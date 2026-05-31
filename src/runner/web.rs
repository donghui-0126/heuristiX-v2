use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::sim::{EngineSnapshot, Metrics, SimulationEngine, StepOutcome};

const INDEX_HTML: &str = include_str!("web_assets/index.html");

#[derive(Serialize, Clone)]
struct WireSnap {
    scenario: String,
    paused: bool,
    finished: bool,
    run_id: u64,
    speed_mul: f64,
    engine: EngineSnapshot,
    metrics: Metrics,
}

#[derive(Default)]
struct ControlState {
    paused: bool,
    speed_mul: f64,
}

#[derive(Deserialize, Default)]
struct ControlPatch {
    paused: Option<bool>,
    speed_mul: Option<f64>,
}

#[derive(Serialize, Clone)]
enum WireEvent {
    Snap(WireSnap),
    Done(WireSnap),
}

type EngineFactory = Arc<dyn Fn() -> SimulationEngine + Send + Sync>;

#[derive(Clone)]
struct AppState {
    tx: broadcast::Sender<WireEvent>,
    control: Arc<Mutex<ControlState>>,
    last_snap: Arc<Mutex<Option<WireSnap>>>,
    abort_flag: Arc<AtomicBool>,
    run_id: Arc<AtomicU64>,
    engine_factory: EngineFactory,
    scenario: String,
    ms_per_sim_minute: u64,
}

pub struct WebRunner {
    engine_factory: EngineFactory,
    scenario: String,
    port: u16,
    ms_per_sim_minute: u64,
}

impl WebRunner {
    pub fn new<F>(engine_factory: F, scenario: String, port: u16, ms_per_sim_minute: u64) -> Self
    where
        F: Fn() -> SimulationEngine + Send + Sync + 'static,
    {
        Self {
            engine_factory: Arc::new(engine_factory),
            scenario,
            port,
            ms_per_sim_minute,
        }
    }

    pub fn run(self) -> Result<()> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(self.run_async())
    }

    async fn run_async(self) -> Result<()> {
        let (tx, _) = broadcast::channel::<WireEvent>(256);
        let control = Arc::new(Mutex::new(ControlState { paused: false, speed_mul: 1.0 }));
        let last_snap: Arc<Mutex<Option<WireSnap>>> = Arc::new(Mutex::new(None));
        let abort_flag = Arc::new(AtomicBool::new(false));
        let run_id = Arc::new(AtomicU64::new(0));

        let state = AppState {
            tx,
            control,
            last_snap,
            abort_flag,
            run_id: run_id.clone(),
            engine_factory: self.engine_factory.clone(),
            scenario: self.scenario.clone(),
            ms_per_sim_minute: self.ms_per_sim_minute,
        };

        spawn_sim_thread(&state, 1);
        run_id.store(1, Ordering::SeqCst);

        let app = Router::new()
            .route("/", get(serve_index))
            .route("/stream", get(sse_stream))
            .route("/control", post(control_post))
            .route("/restart", post(restart_post))
            .with_state(state);

        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], self.port));
        let listener = tokio::net::TcpListener::bind(addr).await?;
        eprintln!("\nweb UI running at http://{}\n", addr);

        let shutdown = async move { std::future::pending::<()>().await };
        axum::serve(listener, app).with_graceful_shutdown(shutdown).await?;
        Ok(())
    }
}

fn spawn_sim_thread(state: &AppState, my_run_id: u64) {
    let tx = state.tx.clone();
    let control = state.control.clone();
    let last_snap = state.last_snap.clone();
    let abort = state.abort_flag.clone();
    let run_id = state.run_id.clone();
    let factory = state.engine_factory.clone();
    let scenario = state.scenario.clone();
    let ms_per_min = state.ms_per_sim_minute;

    std::thread::spawn(move || {
        let mut engine = factory();
        control.lock().unwrap().paused = false;

        // Initial snapshot.
        emit(&tx, &last_snap, &control, &scenario, my_run_id, &mut engine, false, false);

        let walltime_start = Instant::now();
        loop {
            if abort.load(Ordering::SeqCst) || run_id.load(Ordering::SeqCst) != my_run_id {
                return;
            }

            let (paused, mul) = {
                let c = control.lock().unwrap();
                (c.paused, c.speed_mul)
            };
            if paused {
                std::thread::sleep(Duration::from_millis(60));
                emit(&tx, &last_snap, &control, &scenario, my_run_id, &mut engine, true, false);
                continue;
            }

            let target_sim_t = walltime_start.elapsed().as_millis() as f64 * mul / ms_per_min as f64;

            let mut stepped_any = false;
            // Process every event whose time <= target.
            while let Some(t) = engine.next_event_time() {
                if t > target_sim_t { break; }
                match engine.step() {
                    StepOutcome::Done => {
                        emit(&tx, &last_snap, &control, &scenario, my_run_id, &mut engine, false, true);
                        return;
                    }
                    StepOutcome::RenderFrame | StepOutcome::Progressed => {
                        stepped_any = true;
                    }
                }
                if abort.load(Ordering::SeqCst) || run_id.load(Ordering::SeqCst) != my_run_id {
                    return;
                }
            }
            let _ = stepped_any;

            emit(&tx, &last_snap, &control, &scenario, my_run_id, &mut engine, false, false);
            std::thread::sleep(Duration::from_millis(33)); // ~30 fps
        }
    });
}

fn emit(
    tx: &broadcast::Sender<WireEvent>,
    last_snap: &Arc<Mutex<Option<WireSnap>>>,
    control: &Arc<Mutex<ControlState>>,
    scenario: &str,
    run_id: u64,
    engine: &mut SimulationEngine,
    paused: bool,
    finished: bool,
) {
    let snap = engine.snapshot();
    let metrics = engine.finalize();
    let speed_mul = control.lock().unwrap().speed_mul;
    let wire = WireSnap {
        scenario: scenario.to_string(),
        paused,
        finished,
        run_id,
        speed_mul,
        engine: snap,
        metrics,
    };
    *last_snap.lock().unwrap() = Some(wire.clone());
    let _ = if finished {
        tx.send(WireEvent::Done(wire))
    } else {
        tx.send(WireEvent::Snap(wire))
    };
}

async fn serve_index() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], INDEX_HTML)
}

async fn sse_stream(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<SseEvent, std::convert::Infallible>>> {
    use futures::StreamExt;
    let rx = state.tx.subscribe();

    let initial = state.last_snap.lock().unwrap().clone();
    let initial_stream = futures::stream::iter(initial.into_iter().map(|s| {
        Ok(SseEvent::default().event("snap").data(serde_json::to_string(&s).unwrap()))
    }));

    let live = tokio_stream::wrappers::BroadcastStream::new(rx).filter_map(|res| async move {
        match res {
            Ok(WireEvent::Snap(s)) => Some(Ok(SseEvent::default()
                .event("snap").data(serde_json::to_string(&s).unwrap()))),
            Ok(WireEvent::Done(s)) => Some(Ok(SseEvent::default()
                .event("done").data(serde_json::to_string(&s).unwrap()))),
            Err(_) => None,
        }
    });

    Sse::new(initial_stream.chain(live)).keep_alive(KeepAlive::default())
}

async fn control_post(
    State(state): State<AppState>,
    Json(patch): Json<ControlPatch>,
) -> impl IntoResponse {
    let mut c = state.control.lock().unwrap();
    if let Some(p) = patch.paused { c.paused = p; }
    if let Some(s) = patch.speed_mul { c.speed_mul = s.clamp(0.01, 10000.0); }
    StatusCode::OK
}

async fn restart_post(State(state): State<AppState>) -> impl IntoResponse {
    state.abort_flag.store(true, Ordering::SeqCst);
    let new_id = state.run_id.fetch_add(1, Ordering::SeqCst) + 1;
    tokio::time::sleep(Duration::from_millis(80)).await;
    state.abort_flag.store(false, Ordering::SeqCst);
    spawn_sim_thread(&state, new_id);
    StatusCode::OK
}
