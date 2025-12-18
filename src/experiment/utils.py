def create_index_name(experiment_name: str, model_name: str) -> str:
    """Creates a descriptive name for the index directory."""
    sanitized_model_name: str = model_name.replace("/", "_")
    return f"{experiment_name}_{sanitized_model_name}"
