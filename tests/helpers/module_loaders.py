import importlib.util
import itertools
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEECH_ROOT = REPO_ROOT / "openwillis" / "openwillis-speech" / "src" / "openwillis" / "speech"
CHARACTERISTICS_UTIL_PATH = SPEECH_ROOT / "util" / "characteristics_util.py"
SPEECH_ATTRIBUTE_PATH = SPEECH_ROOT / "speech_attribute.py"
PAUSE_MODULE_PATH = SPEECH_ROOT / "util" / "speech" / "pause.py"
LEXICAL_MODULE_PATH = SPEECH_ROOT / "util" / "speech" / "lexical.py"
COHERENCE_MODULE_PATH = SPEECH_ROOT / "util" / "speech" / "coherence.py"

_PACKAGE_COUNTER = itertools.count()


def _stub_nltk_module():
    nltk_module = types.ModuleType("nltk")

    def sent_tokenize(text):
        text = (text or "").strip()
        if not text:
            return []
        sentences = [part.strip() for part in text.replace("!", ".").replace("?", ".").split(".") if part.strip()]
        return sentences or [text]

    nltk_module.data = types.SimpleNamespace(find=lambda *args, **kwargs: True)
    nltk_module.download = lambda *args, **kwargs: True
    nltk_module.tokenize = types.SimpleNamespace(sent_tokenize=sent_tokenize)
    return nltk_module


def _stub_spacy_module():
    spacy_module = types.ModuleType("spacy")
    spacy_module.load = lambda *args, **kwargs: object()
    return spacy_module


def _stub_transformers_module():
    transformers_module = types.ModuleType("transformers")

    class _StubPretrained:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubPipeline:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    transformers_module.AutoTokenizer = _StubPretrained
    transformers_module.AutoModelForCausalLM = _StubPretrained
    transformers_module.AutoModelForSequenceClassification = _StubPretrained
    transformers_module.BertTokenizer = _StubPretrained
    transformers_module.BertModel = _StubPretrained
    transformers_module.BertForMaskedLM = _StubPretrained
    transformers_module.XLMRobertaTokenizer = _StubPretrained
    transformers_module.pipeline = lambda *args, **kwargs: _StubPipeline(*args, **kwargs)
    return transformers_module


def _stub_sentence_transformers_module():
    module = types.ModuleType("sentence_transformers")

    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    module.SentenceTransformer = SentenceTransformer
    return module


def _stub_huggingface_hub_module():
    module = types.ModuleType("huggingface_hub")
    module.login = lambda *args, **kwargs: None
    return module


def _stub_vader_module():
    module = types.ModuleType("vaderSentiment")
    submodule = types.ModuleType("vaderSentiment.vaderSentiment")

    class SentimentIntensityAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

    submodule.SentimentIntensityAnalyzer = SentimentIntensityAnalyzer
    module.vaderSentiment = submodule
    sys.modules[submodule.__name__] = submodule
    return module


def _stub_lexicalrichness_module():
    module = types.ModuleType("lexicalrichness")

    class LexicalRichness:
        def __init__(self, *args, **kwargs):
            pass

    module.LexicalRichness = LexicalRichness
    return module


def _stub_torch_module():
    module = types.ModuleType("torch")

    class _Device:
        def __init__(self, name):
            self.type = name

        def __repr__(self):
            return f"device(type={self.type!r})"

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    class _MPS:
        @staticmethod
        def is_available():
            return False

    class _InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

        def __call__(self, fn):
            return fn

    module.device = _Device
    module.cuda = _Cuda()
    module.backends = types.SimpleNamespace(mps=_MPS(), cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)))
    module.float32 = "float32"
    module.dtype = type("dtype", (), {})
    module.Tensor = object
    module.set_float32_matmul_precision = lambda *args, **kwargs: None
    module.inference_mode = lambda: _InferenceMode()
    return module


def _install_heavy_runtime_stubs():
    sys.modules["transformers"] = _stub_transformers_module()
    sys.modules["sentence_transformers"] = _stub_sentence_transformers_module()
    sys.modules["huggingface_hub"] = _stub_huggingface_hub_module()
    sys.modules["vaderSentiment"] = _stub_vader_module()
    sys.modules["lexicalrichness"] = _stub_lexicalrichness_module()
    sys.modules["torch"] = _stub_torch_module()


def _install_global_stubs():
    sys.modules["nltk"] = _stub_nltk_module()
    sys.modules["spacy"] = _stub_spacy_module()


def _build_package_hierarchy(package_name):
    root_pkg = types.ModuleType(package_name)
    root_pkg.__path__ = [str(REPO_ROOT)]

    speech_pkg = types.ModuleType(f"{package_name}.speech")
    speech_pkg.__path__ = [str(SPEECH_ROOT)]

    util_pkg = types.ModuleType(f"{package_name}.speech.util")
    util_pkg.__path__ = [str(SPEECH_ROOT / "util")]

    util_speech_pkg = types.ModuleType(f"{package_name}.speech.util.speech")
    util_speech_pkg.__path__ = [str(SPEECH_ROOT / "util" / "speech")]

    pause_module = types.ModuleType(f"{package_name}.speech.util.speech.pause")
    pause_module.get_pause_feature = lambda json_conf, df_list, *args, **kwargs: df_list

    lexical_module = types.ModuleType(f"{package_name}.speech.util.speech.lexical")
    lexical_module.get_repetitions = lambda df_list, *args, **kwargs: df_list
    lexical_module.get_sentiment = lambda df_list, *args, **kwargs: df_list
    lexical_module.get_pos_tag = lambda df_list, *args, **kwargs: df_list

    coherence_module = types.ModuleType(f"{package_name}.speech.util.speech.coherence")
    coherence_module.get_word_coherence = lambda df_list, *args, **kwargs: df_list
    coherence_module.get_phrase_coherence = lambda df_list, *args, **kwargs: df_list

    root_pkg.speech = speech_pkg
    speech_pkg.util = util_pkg
    util_pkg.speech = util_speech_pkg

    sys.modules[package_name] = root_pkg
    sys.modules[f"{package_name}.speech"] = speech_pkg
    sys.modules[f"{package_name}.speech.util"] = util_pkg
    sys.modules[f"{package_name}.speech.util.speech"] = util_speech_pkg
    sys.modules[pause_module.__name__] = pause_module
    sys.modules[lexical_module.__name__] = lexical_module
    sys.modules[coherence_module.__name__] = coherence_module

    util_speech_pkg.pause = pause_module
    util_speech_pkg.lexical = lexical_module
    util_speech_pkg.coherence = coherence_module

    return speech_pkg, util_pkg


def _load_module(module_name, source_path):
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_local_characteristics_util():
    _install_global_stubs()
    package_name = f"local_openwillis_speech_test_{next(_PACKAGE_COUNTER)}"
    _, util_pkg = _build_package_hierarchy(package_name)
    module_name = f"{package_name}.speech.util.characteristics_util"
    module = _load_module(module_name, CHARACTERISTICS_UTIL_PATH)
    util_pkg.characteristics_util = module
    return module


def load_local_speech_attribute():
    _install_global_stubs()
    package_name = f"local_openwillis_speech_test_{next(_PACKAGE_COUNTER)}"
    _, util_pkg = _build_package_hierarchy(package_name)

    cutil_name = f"{package_name}.speech.util.characteristics_util"
    cutil_module = _load_module(cutil_name, CHARACTERISTICS_UTIL_PATH)
    util_pkg.characteristics_util = cutil_module

    module_name = f"{package_name}.speech.speech_attribute"
    return _load_module(module_name, SPEECH_ATTRIBUTE_PATH)


def load_local_pause_module():
    _install_global_stubs()
    package_name = f"local_openwillis_speech_test_{next(_PACKAGE_COUNTER)}"
    _, util_pkg = _build_package_hierarchy(package_name)
    module_name = f"{package_name}.speech.util.speech.pause"
    module = _load_module(module_name, PAUSE_MODULE_PATH)
    sys.modules[module_name] = module
    util_pkg.speech.pause = module
    return module


def load_local_lexical_module():
    _install_global_stubs()
    _install_heavy_runtime_stubs()
    package_name = f"local_openwillis_speech_test_{next(_PACKAGE_COUNTER)}"
    _, util_pkg = _build_package_hierarchy(package_name)
    module_name = f"{package_name}.speech.util.speech.lexical"
    module = _load_module(module_name, LEXICAL_MODULE_PATH)
    sys.modules[module_name] = module
    util_pkg.speech.lexical = module
    return module


def load_local_coherence_module():
    _install_global_stubs()
    _install_heavy_runtime_stubs()
    package_name = f"local_openwillis_speech_test_{next(_PACKAGE_COUNTER)}"
    _, util_pkg = _build_package_hierarchy(package_name)
    module_name = f"{package_name}.speech.util.speech.coherence"
    module = _load_module(module_name, COHERENCE_MODULE_PATH)
    sys.modules[module_name] = module
    util_pkg.speech.coherence = module
    return module
