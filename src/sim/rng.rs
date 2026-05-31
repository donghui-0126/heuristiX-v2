use rand::rngs::StdRng;
use rand::SeedableRng;

/// CRN-friendly RNG with separated streams for instance generation and disruption.
pub struct RngStreams {
    pub instance: StdRng,
    pub disruption: StdRng,
}

impl RngStreams {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            instance: StdRng::seed_from_u64(splitmix(seed, 0)),
            disruption: StdRng::seed_from_u64(splitmix(seed, 1)),
        }
    }
}

#[inline]
fn splitmix(seed: u64, salt: u64) -> u64 {
    let mut x = seed.wrapping_add(salt.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
