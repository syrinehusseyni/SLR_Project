import pytest

from slrresearch.core.config import Settings, build_model, load_settings
from slrresearch.core.errors import ConfigError


def test_default_provider_is_ollama_with_llama31():
    s = Settings.from_env({})
    assert s.llm_provider == "ollama"
    assert s.model_for("extraction") == "llama3.1:8b"


def test_switching_provider_switches_default_model():
    env = {"OLLAMA_MODEL": "llama3.1:8b", "GROQ_MODEL": "groq-model-x"}
    assert Settings.from_env({**env, "LLM_PROVIDER": "ollama"}).model_for("writer") == "llama3.1:8b"
    assert Settings.from_env({**env, "LLM_PROVIDER": "groq"}).model_for("writer") == "groq-model-x"


def test_unknown_provider_refused():
    with pytest.raises(ConfigError, match="inconnu"):
        Settings.from_env({"LLM_PROVIDER": "openai"})


def test_unknown_role_refused():
    with pytest.raises(ConfigError, match="role inconnu"):
        Settings.from_env({}).model_for("cuisinier")


def test_invalid_timeout_refused():
    with pytest.raises(ConfigError, match="LLM_TIMEOUT"):
        Settings.from_env({"LLM_TIMEOUT": "abc"})


def test_role_override_only_for_active_provider():
    env = {"OLLAMA_MODEL_JUDGE": "mistral:7b", "GROQ_MODEL_JUDGE": "groq-judge", "GROQ_MODEL": "groq-main"}
    ollama = Settings.from_env({**env, "LLM_PROVIDER": "ollama"})
    groq = Settings.from_env({**env, "LLM_PROVIDER": "groq"})
    assert ollama.model_for("judge") == "mistral:7b"
    assert groq.model_for("judge") == "groq-judge"
    assert ollama.model_for("extraction") == "llama3.1:8b"  # les autres roles gardent le defaut


def test_judge_independence_flag():
    same = Settings.from_env({"LLM_PROVIDER": "ollama"})
    assert same.judge_is_independent is False  # un seul modele local = juge non independant
    diff = Settings.from_env({"LLM_PROVIDER": "ollama", "OLLAMA_MODEL_JUDGE": "mistral:7b"})
    assert diff.judge_is_independent is True


def test_groq_without_key_raises_clear_error():
    s = Settings.from_env({"LLM_PROVIDER": "groq"})
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        build_model(s, "screening")


def test_load_settings_reads_dotenv_file_at_call_time(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=groq\nGROQ_API_KEY=abc\nCONTACT_EMAIL=a@b.c\n", encoding="utf-8")
    s = load_settings(env_file, environ={})
    assert (s.llm_provider, s.groq_api_key, s.contact_email) == ("groq", "abc", "a@b.c")
    env_file.write_text("LLM_PROVIDER=ollama\n", encoding="utf-8")  # on change le fichier...
    assert load_settings(env_file, environ={}).llm_provider == "ollama"  # ...c'est pris en compte


def test_environment_variable_overrides_dotenv(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=ollama\n", encoding="utf-8")
    assert load_settings(env_file, environ={"LLM_PROVIDER": "groq"}).llm_provider == "groq"


def test_missing_dotenv_file_uses_defaults(tmp_path):
    assert load_settings(tmp_path / "absent.env", environ={}).llm_provider == "ollama"


def test_build_ollama_model_has_right_name_and_temperature():
    pytest.importorskip("langchain_ollama")
    model = build_model(Settings.from_env({"LLM_PROVIDER": "ollama"}), "extraction")
    assert model.model == "llama3.1:8b"
    assert model.temperature == 0


def test_build_groq_model_with_key():
    pytest.importorskip("langchain_groq")
    s = Settings.from_env({"LLM_PROVIDER": "groq", "GROQ_API_KEY": "gsk_fake", "GROQ_MODEL": "m1"})
    model = build_model(s, "judge")
    assert model.model_name == "m1"
    # ChatGroq remplace 0 par 1e-8 (Groq refuse 0 exact) : on teste "quasi nul"
    assert model.temperature == pytest.approx(0, abs=1e-6)