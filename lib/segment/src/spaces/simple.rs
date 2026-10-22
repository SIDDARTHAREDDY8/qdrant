use common::types::ScoreType;

use super::metric::{Metric, MetricPostProcessing};
#[cfg(target_arch = "x86_64")]
use super::simple_avx::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use super::simple_neon::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::simple_sse::*;
use super::tools::is_length_zero_or_normalized;
use crate::data_types::vectors::{DenseVector, VectorElementType};
use crate::types::Distance;

#[cfg(target_arch = "x86_64")]
pub(crate) const MIN_DIM_SIZE_AVX: usize = 32;

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
))]
pub(crate) const MIN_DIM_SIZE_SIMD: usize = 16;

#[derive(Clone)]
pub struct DotProductMetric;

#[derive(Clone)]
pub struct CosineMetric;

#[derive(Clone)]
pub struct EuclidMetric;

#[derive(Clone)]
pub struct ManhattanMetric;

impl Metric<VectorElementType> for EuclidMetric {
    fn distance() -> Distance {
        Distance::Euclid
    }

    fn similarity(v1: &[VectorElementType], v2: &[VectorElementType]) -> ScoreType {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx")
                && is_x86_feature_detected!("fma")
                && v1.len() >= MIN_DIM_SIZE_AVX
            {
                return unsafe { euclid_similarity_avx(v1, v2) };
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") && v1.len() >= MIN_DIM_SIZE_SIMD {
                return unsafe { euclid_similarity_sse(v1, v2) };
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") && v1.len() >= MIN_DIM_SIZE_SIMD {
                return unsafe { euclid_similarity_neon(v1, v2) };
            }
        }

        euclid_similarity(v1, v2)
    }

    fn preprocess(vector: DenseVector) -> DenseVector {
        vector
    }
}

impl MetricPostProcessing for EuclidMetric {
    fn postprocess(score: ScoreType) -> ScoreType {
        // `similarity` already returns the negative Euclidean distance (not
        // its square), so post-processing only flips the sign back.
        score.abs()
    }
}

impl Metric<VectorElementType> for ManhattanMetric {
    fn distance() -> Distance {
        Distance::Manhattan
    }

    fn similarity(v1: &[VectorElementType], v2: &[VectorElementType]) -> ScoreType {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx")
                && is_x86_feature_detected!("fma")
                && v1.len() >= MIN_DIM_SIZE_AVX
            {
                return unsafe { manhattan_similarity_avx(v1, v2) };
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") && v1.len() >= MIN_DIM_SIZE_SIMD {
                return unsafe { manhattan_similarity_sse(v1, v2) };
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") && v1.len() >= MIN_DIM_SIZE_SIMD {
                return unsafe { manhattan_similarity_neon(v1, v2) };
            }
        }

        manhattan_similarity(v1, v2)
    }

    fn preprocess(vector: DenseVector) -> DenseVector {
        vector
    }
}

impl MetricPostProcessing for ManhattanMetric {
    fn postprocess(score: ScoreType) -> ScoreType {
        score.abs()
    }
}

impl Metric<VectorElementType> for DotProductMetric {
    fn distance() -> Distance {
        Distance::Dot
    }

    fn similarity(v1: &[VectorElementType], v2: &[VectorElementType]) -> ScoreType {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx")
                && is_x86_feature_detected!("fma")
                && v1.len() >= MIN_DIM_SIZE_AVX
            {
                return unsafe { dot_similarity_avx(v1, v2) };
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") && v1.len() >= MIN_DIM_SIZE_SIMD {
                return unsafe { dot_similarity_sse(v1, v2) };
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") && v1.len() >= MIN_DIM_SIZE_SIMD {
                return unsafe { dot_similarity_neon(v1, v2) };
            }
        }

        dot_similarity(v1, v2)
    }

    fn preprocess(vector: DenseVector) -> DenseVector {
        vector
    }
}

impl MetricPostProcessing for DotProductMetric {
    fn postprocess(score: ScoreType) -> ScoreType {
        score
    }
}

/// Equivalent to DotProductMetric with normalization of the vectors in preprocessing.
impl Metric<VectorElementType> for CosineMetric {
    fn distance() -> Distance {
        Distance::Cosine
    }

    fn similarity(v1: &[VectorElementType], v2: &[VectorElementType]) -> ScoreType {
        DotProductMetric::similarity(v1, v2)
    }

    fn preprocess(vector: DenseVector) -> DenseVector {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx")
                && is_x86_feature_detected!("fma")
                && vector.len() >= MIN_DIM_SIZE_AVX
            {
                return unsafe { cosine_preprocess_avx(vector) };
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") && vector.len() >= MIN_DIM_SIZE_SIMD {
                return unsafe { cosine_preprocess_sse(vector) };
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") && vector.len() >= MIN_DIM_SIZE_SIMD
            {
                return unsafe { cosine_preprocess_neon(vector) };
            }
        }

        cosine_preprocess(vector)
    }
}

impl MetricPostProcessing for CosineMetric {
    fn postprocess(score: ScoreType) -> ScoreType {
        score
    }
}

pub fn euclid_similarity(v1: &[VectorElementType], v2: &[VectorElementType]) -> ScoreType {
    // Return the negative Euclidean distance rather than the negative squared
    // distance. The squared form cannot represent tiny distances in `f32`:
    // for a difference of `1e-25`, `(1e-25)^2 = 1e-50` underflows to `0.0`, so
    // distinct tiny vectors collapsed to a zero score and ranked as exact
    // matches (https://github.com/qdrant/qdrant/issues/11287). It also
    // overflows to infinity for large differences (e.g. `(1e20)^2`).
    //
    // The distance is computed with the classic scaling trick in `f64`:
    // divide every difference by the largest absolute difference before
    // squaring, so the squared sum stays within `[1, dim]` regardless of the
    // magnitude of the inputs, then rescale after the square root.
    let max_diff = v1
        .iter()
        .zip(v2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, ScoreType::max);
    if max_diff == 0.0 {
        return -0.0;
    }
    let max_diff = f64::from(max_diff);
    let scaled_sum: f64 = v1
        .iter()
        .zip(v2)
        .map(|(a, b)| {
            let diff = (f64::from(*a) - f64::from(*b)) / max_diff;
            diff * diff
        })
        .sum();
    -((max_diff * scaled_sum.sqrt()) as ScoreType)
}

pub fn manhattan_similarity(v1: &[VectorElementType], v2: &[VectorElementType]) -> ScoreType {
    -v1.iter()
        .zip(v2)
        .map(|(a, b)| (a - b).abs())
        .sum::<ScoreType>()
}

pub fn cosine_preprocess(vector: DenseVector) -> DenseVector {
    let mut length: f32 = vector.iter().map(|x| x * x).sum();
    if is_length_zero_or_normalized(length) {
        return vector;
    }
    length = length.sqrt();
    vector.iter().map(|x| x / length).collect()
}

pub fn dot_similarity(v1: &[VectorElementType], v2: &[VectorElementType]) -> ScoreType {
    v1.iter().zip(v2).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use rand::RngExt;

    use super::*;

    #[test]
    fn test_cosine_preprocessing() {
        let res = <CosineMetric as Metric<VectorElementType>>::preprocess(vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(res, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_euclid_tiny_vectors_do_not_collapse() {
        // Regression test for https://github.com/qdrant/qdrant/issues/11287
        let query: DenseVector = vec![1e-25, 0.0];
        let self_match: DenseVector = vec![1e-25, 0.0];
        let zero: DenseVector = vec![0.0, 0.0];

        let self_score =
            <EuclidMetric as MetricPostProcessing>::postprocess(<EuclidMetric as Metric<
                VectorElementType,
            >>::similarity(
                &query, &self_match
            ));
        let zero_score =
            <EuclidMetric as MetricPostProcessing>::postprocess(<EuclidMetric as Metric<
                VectorElementType,
            >>::similarity(
                &query, &zero
            ));

        assert_eq!(self_score, 0.0);
        // The true distance is `1e-25`, which is representable in `f32`; it
        // must not collapse to `0.0` through the squared-difference path.
        assert_eq!(zero_score, 1e-25);
        // The exact self-match must rank before the zero vector (higher
        // similarity is closer for the raw score).
        assert!(
            <EuclidMetric as Metric<VectorElementType>>::similarity(&query, &self_match)
                > <EuclidMetric as Metric<VectorElementType>>::similarity(&query, &zero)
        );
    }

    #[test]
    fn test_euclid_tiny_vectors_score_threshold() {
        // A `score_threshold` between the two true distances must keep the
        // self-match and drop the zero vector; before the fix both scored
        // `0.0`, so the threshold could not separate them.
        let query: DenseVector = vec![1e-25, 0.0];
        let self_match: DenseVector = vec![1e-25, 0.0];
        let zero: DenseVector = vec![0.0, 0.0];

        let threshold = 5e-26;
        let self_score =
            Distance::Euclid.postprocess_score(
                <EuclidMetric as Metric<VectorElementType>>::similarity(&query, &self_match),
            );
        let zero_score = Distance::Euclid.postprocess_score(<EuclidMetric as Metric<
            VectorElementType,
        >>::similarity(&query, &zero));

        assert!(Distance::Euclid.check_threshold(self_score, threshold));
        assert!(!Distance::Euclid.check_threshold(zero_score, threshold));
    }

    #[test]
    fn test_euclid_large_vectors_do_not_overflow() {
        let v1: DenseVector = vec![1e20, 0.0];
        let v2: DenseVector = vec![0.0, 0.0];
        let score = <EuclidMetric as MetricPostProcessing>::postprocess(<EuclidMetric as Metric<
            VectorElementType,
        >>::similarity(
            &v1, &v2
        ));
        assert_eq!(score, 1e20);
    }

    #[test]
    fn test_euclid_distance_values() {
        let v1: DenseVector = vec![3.0, 4.0];
        let v2: DenseVector = vec![0.0, 0.0];
        let score = <EuclidMetric as MetricPostProcessing>::postprocess(<EuclidMetric as Metric<
            VectorElementType,
        >>::similarity(
            &v1, &v2
        ));
        assert_eq!(score, 5.0);
        assert_eq!(
            <EuclidMetric as MetricPostProcessing>::postprocess(<EuclidMetric as Metric<
                VectorElementType,
            >>::similarity(
                &v1, &v1
            )),
            0.0
        );
    }

    /// If we preprocess a vector multiple times, we expect the same result.
    /// Renormalization should not produce something different.
    #[test]
    fn test_cosine_stable_preprocessing() {
        const DIM: usize = 1500;
        const ATTEMPTS: usize = 100;

        let mut rng = rand::rng();

        for attempt in 0..ATTEMPTS {
            let range = rng.random_range(-2.5..=0.0)..=rng.random_range(0.0..2.5);
            let vector: Vec<_> = (0..DIM).map(|_| rng.random_range(range.clone())).collect();

            // Preprocess and re-preprocess
            let preprocess1 = <CosineMetric as Metric<VectorElementType>>::preprocess(vector);
            let preprocess2: DenseVector =
                <CosineMetric as Metric<VectorElementType>>::preprocess(preprocess1.clone());

            // All following preprocess attempts must be the same
            assert_eq!(
                preprocess1, preprocess2,
                "renormalization is not stable (vector #{attempt})"
            );
        }
    }
}
