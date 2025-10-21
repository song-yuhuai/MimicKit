import enum
import pickle


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that maps deprecated NumPy module paths.

    NumPy 2.0 renamed ``numpy._core`` to ``numpy.core``.  Motion pickle files
    generated with NumPy 2.0+ therefore reference ``numpy._core`` modules, which
    NumPy 1.x cannot import.  By rewriting the module path we can load those
    files without requiring NumPy 2.x at runtime.
    """

    _LEGACY_PREFIX = "numpy._core"
    _TARGET_PREFIX = "numpy.core"

    def find_class(self, module, name):
        if module.startswith(self._LEGACY_PREFIX):
            module = module.replace(self._LEGACY_PREFIX, self._TARGET_PREFIX, 1)
        return super().find_class(module, name)

class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1

def load_motion(file):
    with open(file, "rb") as filestream:
        in_dict = _NumpyCompatUnpickler(filestream).load()

        loop_mode_val = in_dict["loop_mode"]
        fps = in_dict["fps"]
        frames = in_dict["frames"]
        
        loop_mode = LoopMode(loop_mode_val)

        motion_data = Motion(loop_mode=loop_mode,
                             fps=fps,
                             frames=frames)
    return motion_data

class Motion():
    def __init__(self, loop_mode, fps, frames):
        self.loop_mode = loop_mode
        self.fps = fps
        self.frames = frames
        return

    def save(self, out_file):
        with open(out_file, "wb") as out_f:
            out_dict = {
                "loop_mode": self.loop_mode.value,
                "fps": self.fps,
                "frames": self.frames
            }
            pickle.dump(out_dict, out_f)
        return

    def get_length(self):
        num_frames = self.frames.shape[0]
        motion_len = float(num_frames - 1) / self.fps
        return motion_len