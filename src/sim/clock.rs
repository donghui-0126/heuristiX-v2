/// Logical simulation clock. Time unit is **minutes** (f64).
#[derive(Debug, Clone, Copy, Default)]
pub struct SimClock {
    now: f64,
}

impl SimClock {
    pub fn new() -> Self { Self { now: 0.0 } }

    #[inline] pub fn now(&self) -> f64 { self.now }

    #[inline]
    pub fn advance_to(&mut self, t: f64) {
        debug_assert!(t >= self.now, "clock cannot go backwards: {} -> {}", self.now, t);
        self.now = t;
    }
}
