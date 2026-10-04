"""Models the default sweep leaves out, and why.

The registry lists every wrapper that imports. The sweep roster is the registry
minus this table, so "every model" has a written definition that travels with
the code. Each entry carries one reason:

    slow_duplicate      a faster wrapper of the same algorithm is in the roster
    failed_audit        the library's forecasters failed a sanity audit on M3 series
    broken_individual   one model from an otherwise sound library failed the audit
    gpu_duplicate       a CPU-capable twin of the same architecture is in the roster
    cannot_fit          rejects negative or chaotic series, or the dependency is broken
    deferred            timed out or added no information on the audit set
    multivariate_only   skipped automatically on univariate tasks

Usage:
    from forecast_arena.benchmarks.exclusions import get_active_models, EXCLUDED_MODELS

    active = get_active_models()           # roster in this environment
    active = get_active_models(gpu=True)   # the GPU subset of the roster
"""

EXCLUDED_MODELS: dict[str, str] = {
    # slow duplicates of a faster active wrapper
    "DartsXGBoost": "slow_duplicate",
    "DartsLightGBM": "slow_duplicate",
    "NeuralForecastAutoformer": "slow_duplicate",
    "DartsTransformer": "slow_duplicate",
    "DartsRNN": "slow_duplicate",
    "NeuralForecastBiTCN": "slow_duplicate",
    "NeuralForecastVanillaTransformer": "slow_duplicate",
    "PyPOTSFcstTimeLLM": "slow_duplicate",
    "DartsETS": "slow_duplicate",
    "NeuralForecastDeepAR": "slow_duplicate",
    "NeuralForecastNBEATSx": "slow_duplicate",
    "SktimeSTLForecaster": "slow_duplicate",
    "NeuralForecastKAN": "slow_duplicate",
    "NGBoost": "slow_duplicate",
    "TSFreshForecaster": "slow_duplicate",
    "NeuralForecastDilatedRNN": "slow_duplicate",
    "NeuralForecastGRU": "slow_duplicate",
    "NeuralForecastTCN": "slow_duplicate",
    "NeuralForecastDeepNPTS": "slow_duplicate",
    "NeuralForecastPatchTST": "slow_duplicate",
    "GluonTSTFT": "slow_duplicate",
    "GluonTSPatchTST": "slow_duplicate",
    "GluonTSSimpleFeedForward": "slow_duplicate",
    "SktimeProphet": "slow_duplicate",
    # libraries whose forecasters failed the sanity audit
    "PyPOTSFcstTransformer": "failed_audit",
    "PyPOTSFcstTimesNet": "failed_audit",
    "PyPOTSFcstModernTCN": "failed_audit",
    "PyPOTSSAITS": "failed_audit",
    "PyPOTSFcstSegRNN": "failed_audit",
    "PyPOTSFcstDLinear": "failed_audit",
    "PyPOTSFcstTimeMixer": "failed_audit",
    "PyPOTSFcstMICN": "failed_audit",
    "PyPOTSFcstFITS": "failed_audit",
    "PyPOTSFcstTEFN": "failed_audit",
    "PyPOTSFcstFiLM": "failed_audit",
    "PyPOTSGPVAE": "failed_audit",
    "PyPOTSImputeFormer": "failed_audit",
    "PyPOTSBRITS": "failed_audit",
    "PyPOTSCSDI": "failed_audit",
    "PyPOTSFcstCSDI": "failed_audit",
    "PyPOTSFcstBTTF": "failed_audit",
    "TsaiROCKET": "failed_audit",
    "TsaiPatchTST": "failed_audit",
    "TsaiTST": "failed_audit",
    "TsaiFCN": "failed_audit",
    "TsaiOmniScaleCNN": "failed_audit",
    "TsaiResCNN": "failed_audit",
    "TsaiResNet": "failed_audit",
    "TsaimWDN": "failed_audit",
    "TsaiInceptionTime": "failed_audit",
    "TsaiLSTMFCN": "failed_audit",
    "TsaiXceptionTime": "failed_audit",
    "PyTorchForecastingDeepAR": "failed_audit",
    "PyTorchForecastingRecurrentNetwork": "failed_audit",
    "PyTorchForecastingNBeats": "failed_audit",
    "AutoGluonForecaster": "failed_audit",
    # single models from otherwise sound libraries
    "NeuralForecastDLinear": "broken_individual",
    "NeuralForecastTiDE": "broken_individual",
    "TCN": "broken_individual",
    "DartsTSMixer": "broken_individual",
    "GluonTSWaveNet": "broken_individual",
    "SVR": "broken_individual",
    "GluonTSSeasonalNaive": "broken_individual",
    "TFT": "broken_individual",
    "BlockRNN": "broken_individual",
    # GPU-only twins of architectures the roster already has on CPU
    "NeuralForecastNBEATS": "gpu_duplicate",
    "NeuralForecastNHITS": "gpu_duplicate",
    "NeuralForecastNLinear": "gpu_duplicate",
    "NeuralForecastLSTM": "gpu_duplicate",
    "NeuralForecastRNN": "gpu_duplicate",
    "NeuralForecastMLP": "gpu_duplicate",
    "NeuralForecastTFT": "gpu_duplicate",
    "NeuralForecastTimesNet": "gpu_duplicate",
    "NeuralForecastxLSTM": "gpu_duplicate",
    # deferred on the audit set
    "AutoTS": "deferred",
    "FLAMLForecaster": "deferred",
    "GluonTSMQF2": "deferred",
    "NeuralForecastFEDformer": "deferred",
    "NeuralForecastInformer": "deferred",
    "SktimeTBATS": "deferred",
    # cannot fit the task families or the dependency is broken
    "SktimeTheta": "cannot_fit",
    "StatsForecastAutoCES": "cannot_fit",
    "OrbitLGT": "cannot_fit",
    "TimesFM": "cannot_fit",
    "DeepAR": "cannot_fit",
    "TsaiMiniRocket": "cannot_fit",
    # multivariate only
    "DartsVARIMA": "multivariate_only",
    "VAR": "multivariate_only",
    "NeuralForecastMLPMultivariate": "multivariate_only",
    "NeuralForecastRMoK": "multivariate_only",
    "NeuralForecastSOFTS": "multivariate_only",
    "NeuralForecastStemGNN": "multivariate_only",
    "NeuralForecastTSMixer": "multivariate_only",
    "NeuralForecastTSMixerx": "multivariate_only",
    "NeuralForecastTimeMixer": "multivariate_only",
    "NeuralForecastTimeXer": "multivariate_only",
    "NeuralForecastiTransformer": "multivariate_only",
}

# Models that need, or run far faster on, a GPU.
GPU_MODELS = {
    "Chronos", "Chronos2", "FlowState", "LagLlama", "MOMENT",
    "Moirai", "Moirai2", "MoiraiMoE",
    "TTMFinetuned", "TimeMoE", "Timer", "TimesFM25",
    "TinyTimeMixer", "Toto",
    "NBEATS", "NHiTS", "NLinear", "DLinear", "TiDE",
}


def get_active_models(gpu: bool | None = None) -> list[str]:
    """Return the sorted sweep roster: registered models minus ``EXCLUDED_MODELS``.

    Args:
        gpu: None for the whole roster, True for the GPU subset, False for the CPU subset.
    """
    from forecast_arena.forecasters import list_models

    active = [m for m in list_models() if m not in EXCLUDED_MODELS]
    if gpu is True:
        active = [m for m in active if m in GPU_MODELS]
    elif gpu is False:
        active = [m for m in active if m not in GPU_MODELS]
    return sorted(active)


__all__ = ["EXCLUDED_MODELS", "GPU_MODELS", "get_active_models"]
