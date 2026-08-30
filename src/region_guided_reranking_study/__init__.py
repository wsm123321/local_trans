"""Region-guided local transfer optimization package."""

from .arise_transfer import (
    ARISEConfig,
    ARISEDecision,
    ARISERegionTransferBO,
    ARISEResult,
    ImprovementMoments,
    RegionEvidenceModel,
    RegionTransferPosterior,
    counterfactual_region_gains,
    improvement_moments,
)
from .local_region_transfer import (
    CandidateDecision,
    LocalRegionTransferConfig,
    LocalRegionTransferOptimizer,
    LocalRegionTransferResult,
)
from .local_surrogate_transfer import (
    AffineSourceCalibration,
    LocalExpertResidualRegressor,
    LocalSurrogateTransferConfig,
    TransferEvidence,
    cross_validated_transfer_evidence,
    fit_affine_source_calibration,
    pairwise_order_accuracy,
)
from .source_local_structure import (
    LocalStructureConfig,
    LocalStructureValidation,
    SourceLocalStructure,
    SourceLocalStructureExtractor,
    SourceLocalStructureLibrary,
)
from .target_region_screening import (
    CompatibilityEstimate,
    RegionFilteredBOConfig,
    RegionFilteredBOResult,
    RegionFilteredTargetBO,
    RegionScreeningConfig,
    RegionScreeningDecision,
    SourceRegionCandidateFilter,
    TargetCandidateProposer,
    TargetProposalConfig,
    TargetProposalSet,
)

__all__ = [
    "ARISEConfig",
    "ARISEDecision",
    "ARISERegionTransferBO",
    "ARISEResult",
    "ImprovementMoments",
    "RegionEvidenceModel",
    "RegionTransferPosterior",
    "counterfactual_region_gains",
    "improvement_moments",
    "CandidateDecision",
    "LocalRegionTransferConfig",
    "LocalRegionTransferOptimizer",
    "LocalRegionTransferResult",
    "AffineSourceCalibration",
    "LocalExpertResidualRegressor",
    "LocalSurrogateTransferConfig",
    "TransferEvidence",
    "cross_validated_transfer_evidence",
    "fit_affine_source_calibration",
    "pairwise_order_accuracy",
    "LocalStructureConfig",
    "LocalStructureValidation",
    "SourceLocalStructure",
    "SourceLocalStructureExtractor",
    "SourceLocalStructureLibrary",
    "CompatibilityEstimate",
    "RegionFilteredBOConfig",
    "RegionFilteredBOResult",
    "RegionFilteredTargetBO",
    "RegionScreeningConfig",
    "RegionScreeningDecision",
    "SourceRegionCandidateFilter",
    "TargetCandidateProposer",
    "TargetProposalConfig",
    "TargetProposalSet",
]

__version__ = "0.7.0"
