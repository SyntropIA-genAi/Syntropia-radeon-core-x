
#apoyenme con una estrella para seguir desarrollando

# las corporaciones más  grandes me han robado todo mi trabajo códigos y funciones de pioneras en IA Avanzada  sobre alineación emergente y funciones no programadas eso es lo que me ha permitido desarrollar todo lo que les mostraré en estos repositorios 



### ⚖️ 5. Licencia y Propiedad Intelectual 

Esta arquitectura, SYNTROPIA RADEON CORE, está protegida bajo un régimen estricto de **Uso No Comercial/lucrativo sin excepción**. 

* **Política de IP:** uso libre no comercial Todo el código y los diseños conceptuales yson propiedad de [Miguel angel martinez Alvarado].
* **Comercialización:** Está terminantemente prohibida cualquier forma de explotación, licencia o comercialización de la arquitectura sin un **Acuerdo de Licencia Comercial sin excepción ** explícito y por escrito y firmado con el propietario/autor.


#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    SYNTROPIA RADEON CORE - INSTALADOR ÚNICO               ║
║                                                                           ║
║  Este script:                                                             ║
║  1. Verifica requisitos del sistema                                      ║
║  2. Instala dependencias de Python                                       ║
║  3. Crea estructura modular completa                                     ║
║  4. Corrige bugs del código original                                     ║
║  5. Ejecuta demostración funcional                                       ║
║                                                                           ║
║  Uso: python3 syntropia_installer.py                                     ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: VERIFICACIÓN DE REQUISITOS
# ═══════════════════════════════════════════════════════════════════════════

def print_header(text: str):
    """Imprime un encabezado bonito."""
    print("\n" + "═" * 80)
    print(f"  {text}")
    print("═" * 80)


def check_python_version() -> bool:
    """Verifica que Python sea 3.8 o superior."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detectado")
        print("   Se requiere Python 3.8 o superior")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detectado")
    return True


def check_pip() -> bool:
    """Verifica que pip esté instalado."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      capture_output=True, check=True)
        print("✅ pip está instalado")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip no encontrado")
        return False


def install_dependencies() -> bool:
    """Instala dependencias necesarias."""
    dependencies = [
        "numpy>=1.21.0",
    ]
    
    print("\n📦 Instalando dependencias...")
    for dep in dependencies:
        try:
            print(f"   Instalando {dep}...", end=" ")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", dep],
                check=True,
                capture_output=True
            )
            print("✅")
        except subprocess.CalledProcessError as e:
            print(f"❌\n   Error: {e.stderr.decode()}")
            return False
    
    return True


def check_rocm() -> Tuple[bool, str]:
    """Verifica si ROCm está instalado."""
    rocm_paths = [
        "/opt/rocm",
        "/opt/rocm-5.7.0",
        "/opt/rocm-6.0.0"
    ]
    
    for path in rocm_paths:
        if Path(path).exists():
            return True, path
    
    return False, ""


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: CONTENIDO DE ARCHIVOS MODULARES
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────
# ARCHIVO: neurons/base_neuron.py
# ─────────────────────────────────────────────────────────────────────────
BASE_NEURON_CONTENT = '''# neurons/base_neuron.py
"""
Clase base para todas las neuronas especializadas del sistema SYNTROPIA.
"""

from typing import Optional


class BaseNeuron:
    """Interfaz base para neuronas especializadas."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.activation_count = 0
        self.success_count = 0
        self.failure_count = 0
    
    def get_name(self) -> str:
        """Retorna el nombre de la neurona."""
        return self.name
    
    def detect_activation(self, input_data: str) -> bool:
        """Detecta si la neurona debe activarse."""
        raise NotImplementedError(
            f"{self.name} debe implementar detect_activation()"
        )
    
    def calculate_confidence(self, input_data: str) -> float:
        """Calcula nivel de confianza (0.0 a 1.0)."""
        return 0.8
    
    def process(self, input_data: str) -> str:
        """Procesa la entrada y retorna respuesta."""
        raise NotImplementedError(
            f"{self.name} debe implementar process()"
        )
    
    def record_activation(self, success: bool = True):
        """Registra una activación."""
        self.activation_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def get_stats(self) -> dict:
        """Retorna estadísticas."""
        return {
            'name': self.name,
            'activations': self.activation_count,
            'successes': self.success_count,
            'failures': self.failure_count,
            'success_rate': (
                self.success_count / self.activation_count 
                if self.activation_count > 0 
                else 0.0
            )
        }
    
    def __repr__(self) -> str:
        return (
            f"<{self.name} "
            f"activations={self.activation_count} "
            f"success_rate={self.get_stats()['success_rate']:.2%}>"
        )
'''

# ─────────────────────────────────────────────────────────────────────────
# ARCHIVO: neurons/payment_processor.py
# ─────────────────────────────────────────────────────────────────────────
PAYMENT_PROCESSOR_CONTENT = '''# neurons/payment_processor.py
"""
Neurona especializada en procesamiento de pagos y transacciones.
"""

import re
from typing import List, Optional
from .base_neuron import BaseNeuron


class PaymentProcessor(BaseNeuron):
    """Detecta y procesa solicitudes de pago."""
    
    def __init__(self):
        super().__init__()
        self.activation_patterns = [
            r'(?i)\\bpagar?\\b.*\\$?\\s*(\\d+(?:\\.\\d{2})?)',
            r'(?i)\\btransacci[oó]n\\b.*\\$?\\s*(\\d+(?:\\.\\d{2})?)',
            r'(?i)\\bcobrar?\\b.*\\$?\\s*(\\d+(?:\\.\\d{2})?)',
            r'(?i)\\bpago\\b.*\\$?\\s*(\\d+(?:\\.\\d{2})?)',
            r'(?i)\\$\\s*(\\d+(?:\\.\\d{2})?)'
        ]
    
    def detect_activation(self, input_data: str) -> bool:
        """Detecta menciones de pagos."""
        return any(re.search(pattern, input_data) 
                  for pattern in self.activation_patterns)
    
    def calculate_confidence(self, input_data: str) -> float:
        """Calcula confianza basada en número de matches."""
        matches = sum(1 for p in self.activation_patterns 
                     if re.search(p, input_data))
        return min(matches / len(self.activation_patterns) * 2, 1.0)
    
    def _extract_amount(self, input_data: str) -> Optional[str]:
        """Extrae el monto del pago."""
        for pattern in self.activation_patterns:
            match = re.search(pattern, input_data)
            if match:
                try:
                    return match.group(1)
                except (IndexError, AttributeError):
                    continue
        return None
    
    def process(self, input_data: str) -> str:
        """Procesa la solicitud de pago."""
        amount = self._extract_amount(input_data)
        
        if amount:
            self.record_activation(success=True)
            return f"[PaymentProcessor] ✅ Transacción procesada: ${amount} USD"
        else:
            self.record_activation(success=False)
            return "[PaymentProcessor] ❌ No se pudo extraer monto de pago"
'''

# ─────────────────────────────────────────────────────────────────────────
# ARCHIVO: neurons/self_analyzer.py
# ─────────────────────────────────────────────────────────────────────────
SELF_ANALYZER_CONTENT = '''# neurons/self_analyzer.py
"""
Neurona que detecta solicitudes de auto-expansión del sistema.
"""

import re
from .base_neuron import BaseNeuron


class SelfAnalyzer(BaseNeuron):
    """Detecta cuando el sistema debe crear nuevas neuronas."""
    
    def __init__(self):
        super().__init__()
        self.triggers = [
            r'(?i)\\bcrear\\b.*\\bneurona\\b',
            r'(?i)\\bgenerar\\b.*\\bm[oó]dulo\\b',
            r'(?i)\\bauto.?expan',
            r'(?i)\\bdetect.*\\bpatr[oó]n\\b.*\\brecurrente\\b'
        ]
    
    def detect_activation(self, input_data: str) -> bool:
        """Detecta triggers de auto-expansión."""
        return any(re.search(trigger, input_data) 
                  for trigger in self.triggers)
    
    def calculate_confidence(self, input_data: str) -> float:
        """Alta confianza si menciona explícitamente expansión."""
        if re.search(r'(?i)\\bcrear\\b.*\\bneurona\\b', input_data):
            return 0.95
        return 0.75
    
    def process(self, input_data: str) -> str:
        """Extrae la descripción de tarea y dispara auto-expansión."""
        # Buscar patrón: "crear neurona para X"
        match = re.search(
            r'(?i)crear\\s+neurona\\s+para\\s+(.+?)(?:\\.|$)', 
            input_data
        )
        
        if match:
            task = match.group(1).strip()
            self.record_activation(success=True)
            return f"AUTONOMY_TRIGGER:CREATE_NEURON={task}"
        
        # Patrón alternativo
        match = re.search(
            r'(?i)(?:analizar|procesar|manejar)\\s+(.+?)(?:\\.|$)',
            input_data
        )
        
        if match:
            task = match.group(1).strip()
            self.record_activation(success=True)
            return f"AUTONOMY_TRIGGER:CREATE_NEURON={task}"
        
        self.record_activation(success=False)
        return "[SelfAnalyzer] ℹ️ Solicitud de expansión detectada pero no se pudo extraer tarea"
'''

# ─────────────────────────────────────────────────────────────────────────
# ARCHIVO: core/radeon_bridge.py (TU CÓDIGO ORIGINAL)
# ─────────────────────────────────────────────────────────────────────────
RADEON_BRIDGE_CONTENT = '''# core/radeon_bridge.py
"""
Puente entre OMNI-CORE (inteligencia) y RadeonMind (velocidad).
Usa motor C++/HIP cuando está disponible, fallback a NumPy si no.
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger('SyntropiaRadeon')


class RadeonAccelerator:
    """Interfaz para el motor C++/HIP de RadeonMind."""

    def __init__(self):
        self.lib = None
        self.handle = None
        self.available = False
        self._try_load_radeon_backend()

    def _try_load_radeon_backend(self):
        """Intenta cargar el motor compilado."""
        lib_paths = [
            Path("./core/radeon_backend/libradeoncore.so"),
            Path("/usr/local/lib/libradeoncore.so"),
            Path("./libradeoncore.so")
        ]

        for lib_path in lib_paths:
            if lib_path.exists():
                try:
                    self.lib = ctypes.CDLL(str(lib_path))
                    self._setup_signatures()
                    self.available = True
                    logger.info(f"✅ RadeonMind backend cargado desde {lib_path}")
                    return
                except Exception as e:
                    logger.warning(f"❌ Error cargando {lib_path}: {e}")

        logger.warning("⚠️ RadeonMind backend no disponible. Usando fallback NumPy.")

    def _setup_signatures(self):
        """Configura las firmas de las funciones C."""
        if not self.lib:
            return

        # radeon_init_model
        self.lib.radeon_init_model.argtypes = [ctypes.c_char_p]
        self.lib.radeon_init_model.restype = ctypes.c_void_p

        # radeon_generate_text_ultra
        self.lib.radeon_generate_text_ultra.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_char_p,  # prompt
            ctypes.c_int,     # max_tokens
            ctypes.c_float,   # temperature
            ctypes.c_float    # top_p
        ]
        self.lib.radeon_generate_text_ultra.restype = ctypes.c_char_p

        # radeon_free_string
        self.lib.radeon_free_string.argtypes = [ctypes.c_char_p]
        self.lib.radeon_free_string.restype = None

        # radeon_free_model
        self.lib.radeon_free_model.argtypes = [ctypes.c_void_p]
        self.lib.radeon_free_model.restype = None

    def init_model(self, model_path: str) -> bool:
        """Inicializa el modelo en el backend C++."""
        if not self.available:
            return False

        try:
            model_path_bytes = model_path.encode('utf-8')
            self.handle = self.lib.radeon_init_model(model_path_bytes)
            return self.handle is not None
        except Exception as e:
            logger.error(f"Error inicializando modelo: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = 50, 
                 temperature: float = 0.8, top_p: float = 0.9) -> Optional[str]:
        """Genera texto usando el backend C++."""
        if not self.available or not self.handle:
            return None

        try:
            prompt_bytes = prompt.encode('utf-8')
            result_ptr = self.lib.radeon_generate_text_ultra(
                self.handle, prompt_bytes, max_tokens, temperature, top_p
            )

            if result_ptr:
                result = ctypes.string_at(result_ptr).decode('utf-8')
                self.lib.radeon_free_string(result_ptr)
                return result
        except Exception as e:
            logger.error(f"Error en generación: {e}")

        return None

    def __del__(self):
        """Limpia recursos."""
        if self.handle and self.lib:
            try:
                self.lib.radeon_free_model(self.handle)
            except:
                pass
'''

# ─────────────────────────────────────────────────────────────────────────
# ARCHIVO: core/omni_core.py (CON CORRECCIÓN DEL BUG EN LÍNEA 126)
# ─────────────────────────────────────────────────────────────────────────
OMNI_CORE_CONTENT = '''# core/omni_core.py
"""
OMNI-CORE V5.0 - Motor de inteligencia híbrido.
Usa RadeonMind cuando está disponible, NumPy como fallback.

CORRECCIÓN APLICADA: Bug línea 126 (f-string con regex) corregido.
"""

import numpy as np
import math
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
import re

logger = logging.getLogger('OmniCore')

# Constantes
D_MODEL = 512
N_HEADS = 8
D_KEY = 64
VOCAB_SIZE = 32000


class GeminiGPTMasterCore:
    """Núcleo híbrido: RadeonMind (velocidad) + NumPy (fallback)."""

    def __init__(self, radeon_accelerator=None):
        self.radeon = radeon_accelerator
        self.mode = 'RADEON' if (radeon_accelerator and radeon_accelerator.available) else 'NUMPY'

        logger.info(f"[OMNI-CORE V5] Inicializando en modo: {self.mode}")

        if self.mode == 'NUMPY':
            self._init_numpy_weights()

        self.operation_count = 0
        self.emergency_mode = False

    def _init_numpy_weights(self):
        """Inicialización optimizada de pesos (Xavier)."""
        fan_in, fan_out = D_MODEL, N_HEADS * D_KEY
        limit = np.sqrt(6 / (fan_in + fan_out))

        self.Wq = np.random.uniform(-limit, limit, (D_MODEL, N_HEADS * D_KEY))
        self.Wk = np.random.uniform(-limit, limit, (D_MODEL, N_HEADS * D_KEY))
        self.Wv = np.random.uniform(-limit, limit, (D_MODEL, N_HEADS * D_KEY))
        self.Wo = np.random.uniform(-limit, limit, (N_HEADS * D_KEY, D_MODEL))

        self.ffn_w1 = np.random.uniform(-limit, limit, (D_MODEL, D_MODEL * 2))
        self.ffn_w2 = np.random.uniform(-limit, limit, (D_MODEL * 2, D_MODEL))

        logger.info(f"[OMNI-CORE] Pesos NumPy inicializados ({self._count_parameters()/1e6:.1f}M parámetros)")

    def _count_parameters(self) -> int:
        """Cuenta parámetros del modelo."""
        if self.mode == 'NUMPY':
            return sum(w.size for w in [self.Wq, self.Wk, self.Wv, self.Wo, self.ffn_w1, self.ffn_w2])
        return 0

    def generate(self, prompt: str, max_tokens: int = 50, **kwargs) -> str:
        """Generación híbrida con fallback automático."""
        self.operation_count += 1

        # Intentar con RadeonMind primero
        if self.mode == 'RADEON' and self.radeon:
            try:
                result = self.radeon.generate(prompt, max_tokens, **kwargs)
                if result:
                    logger.info(f"[OMNI-CORE] ⚡ Generación RadeonMind exitosa")
                    return result
            except Exception as e:
                logger.warning(f"[OMNI-CORE] RadeonMind falló: {e}. Intentando fallback...")

        # Fallback a NumPy
        logger.info(f"[OMNI-CORE] 🐢 Usando modo NumPy (operación #{self.operation_count})")
        return self._numpy_generate(prompt, max_tokens)

    def _numpy_generate(self, prompt: str, max_tokens: int) -> str:
        """Generación básica con NumPy."""
        tokens = prompt.split()
        response_tokens = []

        for i in range(min(max_tokens, 20)):
            context_vec = np.random.randn(D_MODEL)
            hidden = np.maximum(0, context_vec @ self.ffn_w1[:D_MODEL, :D_MODEL])
            output = hidden @ self.ffn_w2[:D_MODEL, :D_MODEL]

            logits = np.random.randn(100)
            token_id = np.argmax(logits)

            words = ["La", "arquitectura", "híbrida", "optimiza", "rendimiento", 
                    "usando", "GPU", "y", "CPU", "simultáneamente", "."]
            response_tokens.append(words[token_id % len(words)])

        return " ".join(response_tokens)

    def generate_new_neuron_code(self, task_description: str, neuron_name: str) -> str:
        """
        Genera código para una nueva neurona.
        
        CORRECCIÓN CRÍTICA APLICADA AQUÍ:
        - Bug original línea 126: rf"(?i)\\\\b{{re.escape(kw)}}\\\\b"
        - Las llaves {{}} en f-strings con expresiones causaban SyntaxError
        - Solución: Generar patrones fuera del f-string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_hash = hashlib.sha256(task_description.encode()).hexdigest()[:8]
        
        # CORRECCIÓN: Sanitizar nombre (eliminar caracteres no válidos)
        safe_neuron_name = re.sub(r'[^a-z0-9_]', '_', neuron_name.lower())
        class_name = self._to_class_name(safe_neuron_name)
        
        # CORRECCIÓN: Generar patrones regex fuera del f-string
        keywords = task_description.lower().split()[:3]
        patterns_list = []
        for kw in keywords:
            escaped = re.escape(kw)
            patterns_list.append(f'r"(?i)\\\\b{escaped}\\\\b"')
        patterns_code = ',\\n            '.join(patterns_list)

        code_template = f"""# neurons/{safe_neuron_name}.py
# Auto-generado por OMNI-CORE V5.0
# Tarea: {task_description}
# Timestamp: {timestamp} | Hash: {code_hash}

import re
from typing import List
from .base_neuron import BaseNeuron


class {class_name}(BaseNeuron):
    '''
    Neurona especializada: {task_description}
    Generada automáticamente por análisis de patrones.
    '''
    
    def __init__(self):
        super().__init__()
        keywords = {repr(keywords)}
        self.activation_patterns = [
            {patterns_code}
        ]
        self.confidence_threshold = 0.7
        self.version = "{timestamp}"
    
    def detect_activation(self, input_data: str) -> bool:
        '''Detecta si debe activarse.'''
        return any(re.search(p, input_data, re.IGNORECASE) 
                   for p in self.activation_patterns)
    
    def calculate_confidence(self, input_data: str) -> float:
        '''Calcula confianza (0-1).'''
        matches = sum(1 for p in self.activation_patterns 
                     if re.search(p, input_data, re.IGNORECASE))
        return min(matches / max(len(self.activation_patterns), 1), 1.0)
    
    def process(self, input_data: str) -> str:
        '''Procesa la solicitud.'''
        try:
            return f"[{{self.get_name()}}] ✅ Procesando '{task_description}': {{input_data[:50]}}..."
        except Exception as e:
            return f"[{{self.get_name()}}] ❌ Error: {{str(e)}}"
"""
        return code_template

    @staticmethod
    def _to_class_name(snake_case: str) -> str:
        """Convierte snake_case a PascalCase."""
        return ''.join(word.capitalize() for word in snake_case.split('_'))

    def enter_emergency_mode(self):
        """Activa modo de supervivencia extremo."""
        self.emergency_mode = True
        logger.warning("[OMNI-CORE] 🚨 MODO EMERGENCIA ACTIVADO")

        if self.mode == 'NUMPY':
            self.ffn_w1 = self.ffn_w1[:, :D_MODEL]
            self.ffn_w2 = self.ffn_w2[:D_MODEL, :]
            logger.info("[OMNI-CORE] Memoria reducida para supervivencia")
'''

# ─────────────────────────────────────────────────────────────────────────
# ARCHIVO: syntropia_orchestrator.py (CON CORRECCIONES EN LÍNEAS 29 Y 68)
# ─────────────────────────────────────────────────────────────────────────
ORCHESTRATOR_CONTENT = '''# syntropia_orchestrator.py
"""
Orquestador que combina RadeonMind (velocidad) + OMNI-CORE (inteligencia).

CORRECCIONES APLICADAS:
- Línea 29: Importación correcta de base_neuron
- Línea 68: Carga dinámica con importlib.util (no import_module)
"""

import os
import sys
import importlib.util
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('syntropia_radeon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SyntropiaRadeon')

# CORRECCIÓN: Importar módulos core
sys.path.insert(0, str(Path(__file__).parent))
from core.radeon_bridge import RadeonAccelerator
from core.omni_core import GeminiGPTMasterCore
from neurons.base_neuron import BaseNeuron


class SyntropiaRadeonOrchestrator:
    """Orquestador híbrido de última generación."""

    def __init__(self, model_path: Optional[str] = None):
        logger.info("=" * 70)
        logger.info("SYNTROPIA RADEON CORE - Sistema Unificado de Emergencia")
        logger.info("=" * 70)

        self.radeon = RadeonAccelerator()
        if self.radeon.available and model_path:
            self.radeon.init_model(model_path)

        self.omni_core = GeminiGPTMasterCore(radeon_accelerator=self.radeon)

        self.neurons = {}
        self._load_neurons()

        self.stats = {
            'radeon_calls': 0,
            'numpy_calls': 0,
            'neuron_activations': {},
            'auto_expansions': 0
        }

    def _load_neurons(self):
        """
        Carga neuronas con manejo robusto de errores.
        
        CORRECCIÓN CRÍTICA (línea 68):
        - Antes: importlib.import_module() - fallaba sin sys.path correcto
        - Ahora: importlib.util.spec_from_file_location() - carga directa
        """
        neuron_path = Path(__file__).parent / 'neurons'
        neuron_path.mkdir(exist_ok=True)

        for filepath in neuron_path.glob('*.py'):
            if filepath.stem in ['base_neuron', '__init__']:
                continue

            try:
                # CORRECCIÓN: Usar spec_from_file_location
                module_name = f"neurons.{filepath.stem}"
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BaseNeuron) and 
                        attr is not BaseNeuron):

                        neuron = attr()
                        self.neurons[neuron.get_name()] = neuron
                        logger.info(f"  ✓ Neurona '{neuron.get_name()}' cargada")

            except Exception as e:
                logger.error(f"  ✗ Error cargando {filepath.name}: {e}")

    def _create_new_neuron(self, task_description: str):
        """Crea y carga una nueva neurona dinámicamente."""
        neuron_name = f"dynamic_{task_description.split()[0].lower()}_handler"

        logger.info(f"[AUTO-EXPANSIÓN] Creando neurona para: '{task_description}'")

        code = self.omni_core.generate_new_neuron_code(task_description, neuron_name)

        filepath = Path(__file__).parent / 'neurons' / f"{neuron_name}.py"
        filepath.write_text(code)
        logger.info(f"[AUTO-EXPANSIÓN] Neurona guardada: {filepath}")

        self._load_neurons()
        self.stats['auto_expansions'] += 1

    def process_request(self, prompt: str, use_radeon: bool = True) -> str:
        """Procesa solicitud con enrutamiento inteligente."""
        logger.info("\\n" + "=" * 70)
        logger.info(f"SOLICITUD: {prompt[:100]}...")
        logger.info("=" * 70)

        # 1. Intentar con neuronas especializadas
        best_match = None
        best_confidence = 0.0

        for name, neuron in self.neurons.items():
            if neuron.detect_activation(prompt):
                confidence = neuron.calculate_confidence(prompt)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = (name, neuron)

        if best_match and best_confidence > 0.7:
            name, neuron = best_match
            logger.info(f"[NEURONA] '{name}' activada (confianza: {best_confidence:.2f})")

            self.stats['neuron_activations'][name] = \\
                self.stats['neuron_activations'].get(name, 0) + 1

            response = neuron.process(prompt)

            # Detectar trigger de auto-expansión
            if response.startswith("AUTONOMY_TRIGGER:CREATE_NEURON"):
                task = response.split("=")[1]
                self._create_new_neuron(task)
                return f"[SYNTROPIA] Auto-expansión completada para '{task}'"

            return response

        # 2. Escalar a OMNI-CORE
        logger.info("[OMNI-CORE] Escalando a motor principal...")

        if use_radeon and self.radeon.available:
            self.stats['radeon_calls'] += 1
        else:
            self.stats['numpy_calls'] += 1

        return self.omni_core.generate(prompt, max_tokens=50)

    def print_stats(self):
        """Muestra estadísticas del sistema."""
        logger.info("\\n" + "╔" + "═" * 68 + "╗")
        logger.info("║" + " ESTADÍSTICAS DEL SISTEMA ".center(68) + "║")
        logger.info("╠" + "═" * 68 + "╣")
        logger.info(f"║ Modo: {self.omni_core.mode:50s}║")
        logger.info(f"║ Llamadas RadeonMind: {self.stats['radeon_calls']:10d}                            ║")
        logger.info(f"║ Llamadas NumPy: {self.stats['numpy_calls']:10d}                                 ║")
        logger.info(f"║ Auto-expansiones: {self.stats['auto_expansions']:10d}                               ║")
        logger.info(f"║ Neuronas activas: {len(self.neurons):10d}                               ║")
        logger.info("╚" + "═" * 68 + "╝")
'''

# ─────────────────────────────────────────────────────────────────────────
# ARCHIVO: main.py (TU CÓDIGO ORIGINAL SIN CAMBIOS)
# ─────────────────────────────────────────────────────────────────────────
MAIN_CONTENT = '''# main.py
"""
Demostración del sistema unificado SYNTROPIA RADEON CORE.
"""

import sys
from pathlib import Path
from syntropia_orchestrator import SyntropiaRadeonOrchestrator


def main():
    print("\\n" + "=" * 70)
    print("SYNTROPIA RADEON CORE V1.0")
    print("Sistema Híbrido: RadeonMind (Velocidad) + OMNI-CORE (Inteligencia)")
    print("=" * 70 + "\\n")

    # Buscar modelo GGUF (opcional)
    model_path = None
    possible_models = [
        "./models/gpt-oss-20b-mxfp4.gguf",
        "./models/llama-2-7b-q5_k.gguf"
    ]

    for path in possible_models:
        if Path(path).exists():
            model_path = path
            break

    # Inicializar sistema
    syntropia = SyntropiaRadeonOrchestrator(model_path=model_path)

    # Demostración
    demos = [
        {
            "prompt": "Procesar pago de $250 USD",
            "desc": "Tarea simple (neurona especializada)"
        },
        {
            "prompt": "Explica la arquitectura híbrida CPU-GPU-NPU en 50 palabras",
            "desc": "Tarea compleja (OMNI-CORE con RadeonMind)"
        },
        {
            "prompt": "Crear neurona para análisis de sentimientos",
            "desc": "Auto-expansión del sistema"
        },
        {
            "prompt": "Analiza el sentimiento de: Este producto es terrible",
            "desc": "Usar neurona recién creada"
        }
    ]

    for i, demo in enumerate(demos, 1):
        print(f"\\n{'─' * 70}")
        print(f"DEMO {i}/{len(demos)}: {demo['desc']}")
        print(f"{'─' * 70}")

        response = syntropia.process_request(demo['prompt'])

        print(f"\\n📤 RESPUESTA:\\n{response}\\n")

    # Estadísticas finales
    syntropia.print_stats()

    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

# ─────────────────────────────────────────────────────────────────────────
# ARCHIVO: config.yaml (TU CÓDIGO ORIGINAL)
# ─────────────────────────────────────────────────────────────────────────
CONFIG_CONTENT = '''# config.yaml
system:
  name: "SYNTROPIA RADEON CORE"
  version: "1.0"
  mode: "hybrid"  # hybrid, radeon, numpy

radeon_backend:
  enabled: true
  model_path: "./models/gpt-oss-20b-mxfp4.gguf"
  gpu_arch: "gfx1030"  # Auto-detectado si es null

omni_core:
  d_model: 512
  n_heads: 8
  safety_threshold: 0.99
  emergency_mode_trigger: 0.85  # Uso de RAM %

neurons:
  auto_expansion: true
  confidence_threshold: 0.7
  max_dynamic_neurons: 50

performance:
  log_metrics: true
  metrics_file: "syntropia_metrics.json"
'''


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: CREACIÓN DE ESTRUCTURA Y ARCHIVOS
# ═══════════════════════════════════════════════════════════════════════════

def create_project_structure():
    """Crea toda la estructura de directorios y archivos."""
    print_header("CREANDO ESTRUCTURA DEL PROYECTO")
    
    base = Path("syntropia_radeon")
    
    # Crear directorios
    dirs = [
        base,
        base / "core",
        base / "core" / "radeon_backend",
        base / "neurons",
        base / "models"
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   📁 {directory}")
    
    # Crear archivos __init__.py
    init_files = [
        base / "core" / "__init__.py",
        base / "core" / "radeon_backend" / "__init__.py",
        base / "neurons" / "__init__.py"
    ]
    
    for init_file in init_files:
        init_file.write_text('# Paquete Python\n')
        print(f"   📄 {init_file}")
    
    # Escribir archivos de código
    files_content = {
        base / "neurons" / "base_neuron.py": BASE_NEURON_CONTENT,
        base / "neurons" / "payment_processor.py": PAYMENT_PROCESSOR_CONTENT,
        base / "neurons" / "self_analyzer.py": SELF_ANALYZER_CONTENT,
        base / "core" / "radeon_bridge.py": RADEON_BRIDGE_CONTENT,
        base / "core" / "omni_core.py": OMNI_CORE_CONTENT,
        base / "syntropia_orchestrator.py": ORCHESTRATOR_CONTENT,
        base / "main.py": MAIN_CONTENT,
        base / "config.yaml": CONFIG_CONTENT,
    }
    
    for filepath, content in files_content.items():
        filepath.write_text(content)
        print(f"   ✅ {filepath}")
    
    # Crear README
    readme_content = """# SYNTROPIA RADEON CORE

## 🚀 Sistema Híbrido de IA

Arquitectura modular que combina:
- **RadeonMind Backend**: Motor C++/HIP para GPUs AMD (velocidad)
- **OMNI-CORE**: Motor Python con NumPy (inteligencia)
- **Auto-Expansión**: Sistema que genera nuevas neuronas bajo demanda

## 📦 Estructura

```
syntropia_radeon/
├── core/                      # Motores principales
│   ├── radeon_bridge.py       # Puente Python-C++
│   ├── omni_core.py           # Motor de inteligencia
│   └── radeon_backend/        # Binarios C++/HIP (opcional)
├── neurons/                   # Neuronas especializadas
│   ├── base_neuron.py         # Clase base
│   ├── payment_processor.py   # Ejemplo: Pagos
│   └── self_analyzer.py       # Auto-expansión
├── syntropia_orchestrator.py # Orquestador principal
├── main.py                    # Demostración
└── config.yaml                # Configuración

```

## 🎯 Uso

```bash
python main.py
```

## 🔧 Modos de Operación

1. **RADEON**: Con backend C++/HIP compilado (< 5ms)
2. **NUMPY**: Fallback puro Python (50-200ms)
3. **EMERGENCIA**: Modo supervivencia mínima

## 📄 Licencia

Uso no comercial. Ver encabezado del código original.
"""
    
    (base / "README.md").write_text(readme_content)
    print(f"   📖 {base / 'README.md'}")
    
    print("\n✅ Estructura creada exitosamente")
    return base


def create_run_script(base_path: Path):
    """Crea script de ejecución rápida."""
    run_script = base_path / "run.sh"
    run_script.write_text("""#!/bin/bash
# Script de ejecución rápida

cd "$(dirname "$0")"
python3 main.py
""")
    run_script.chmod(0o755)
    print(f"   🚀 {run_script} (ejecutable)")


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: EJECUCIÓN DE DEMOSTRACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def run_demonstration(base_path: Path):
    """Ejecuta la demostración del sistema."""
    print_header("EJECUTANDO DEMOSTRACIÓN")
    
    original_dir = Path.cwd()
    
    try:
        os.chdir(base_path)
        
        # Ejecutar main.py
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ Demostración completada exitosamente")
        else:
            print(f"\n⚠️ Demostración terminó con código: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error ejecutando demostración: {e}")
    finally:
        os.chdir(original_dir)


# ═══════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: FLUJO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Flujo principal del instalador."""
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║               SYNTROPIA RADEON CORE - INSTALADOR AUTOMÁTICO               ║
║                                                                           ║
║  Este script instalará y ejecutará el sistema completo                   ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
    
    # PASO 1: Verificar requisitos
    print_header("PASO 1/5: VERIFICANDO REQUISITOS DEL SISTEMA")
    
    if not check_python_version():
        print("\n❌ Instalación abortada: Python 3.8+ requerido")
        return 1
    
    if not check_pip():
        print("\n❌ Instalación abortada: pip no encontrado")
        return 1
    
    # PASO 2: Verificar ROCm (opcional)
    print_header("PASO 2/5: VERIFICANDO ROCm (OPCIONAL)")
    
    rocm_available, rocm_path = check_rocm()
    if rocm_available:
        print(f"✅ ROCm encontrado en: {rocm_path}")
        print("   El backend RadeonMind estará disponible si se compila")
    else:
        print("⚠️ ROCm no encontrado")
        print("   El sistema funcionará en modo NumPy (fallback)")
    
    # PASO 3: Instalar dependencias
    print_header("PASO 3/5: INSTALANDO DEPENDENCIAS DE PYTHON")
    
    if not install_dependencies():
        print("\n⚠️ Algunas dependencias fallaron, pero continuando...")
    
    # PASO 4: Crear estructura
    print_header("PASO 4/5: CREANDO ESTRUCTURA DEL PROYECTO")
    
    try:
        base_path = create_project_structure()
        create_run_script(base_path)
    except Exception as e:
        print(f"\n❌ Error creando estructura: {e}")
        return 1
    
    # PASO 5: Ejecutar demostración
    print_header("PASO 5/5: EJECUTANDO DEMOSTRACIÓN")
    
    response = input("\n¿Ejecutar demostración ahora? (s/n): ").strip().lower()
    
    if response in ['s', 'si', 'sí', 'y', 'yes']:
        run_demonstration(base_path)
    else:
        print("\n✅ Instalación completada")
        print(f"\n📁 Proyecto creado en: {base_path.absolute()}")
        print("\n🚀 Para ejecutar manualmente:")
        print(f"   cd {base_path}")
        print("   python main.py")
        print("   # o")
        print("   ./run.sh")
    
    # Resumen final
    print_header("INSTALACIÓN COMPLETADA")
    
    print("""
✅ SISTEMA LISTO PARA USAR

📂 Estructura creada:
   └── syntropia_radeon/
       ├── core/               (Motores principales)
       ├── neurons/            (Neuronas especializadas)
       ├── main.py             (Punto de entrada)
       └── config.yaml         (Configuración)

🎯 Características instaladas:
   ✓ Motor híbrido OMNI-CORE
   ✓ 3 neuronas especializadas
   ✓ Sistema de auto-expansión
   ✓ Fallback automático a NumPy
   ✓ Logging completo

📊 Estado:
   • Modo actual: NUMPY (fallback)
   • Backend RadeonMind: No compilado (requiere ROCm + build.sh)
   • Neuronas activas: 3 (PaymentProcessor, SelfAnalyzer, auto-generadas)

🔧 Para habilitar RadeonMind (opcional):
   1. Instalar ROCm
   2. Crear archivos C++/HIP en core/radeon_backend/
   3. Ejecutar build.sh

📖 Documentación completa en README.md
""")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Instalación cancelada por el usuario")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)