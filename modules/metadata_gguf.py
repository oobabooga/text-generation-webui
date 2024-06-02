import os
import sys
import struct
from enum import IntEnum
from io import BufferedReader
from typing import Union

class GGUFValueType(IntEnum):
    # Occasionally check to ensure this class is consistent with gguf
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12


# the GGUF format versions that this module supports
SUPPORTED_GGUF_VERSIONS = [2, 3]

# GGUF only supports execution on little or big endian machines
if sys.byteorder not in ['little', 'big']:
    raise ValueError(
        "host is not little or big endian - GGUF is unsupported"
    )

# arguments for struct.unpack() based on gguf value type
value_packing: dict = {
    GGUFValueType.UINT8:   "=B",
    GGUFValueType.INT8:    "=b",
    GGUFValueType.UINT16:  "=H",
    GGUFValueType.INT16:   "=h",
    GGUFValueType.UINT32:  "=I",
    GGUFValueType.INT32:   "=i",
    GGUFValueType.FLOAT32: "=f",
    GGUFValueType.UINT64:  "=Q",
    GGUFValueType.INT64:   "=q",
    GGUFValueType.FLOAT64: "=d",
    GGUFValueType.BOOL:    "?"
}

# length in bytes for each gguf value type
value_lengths: dict = {
    GGUFValueType.UINT8:   1,
    GGUFValueType.INT8:    1,
    GGUFValueType.UINT16:  2,
    GGUFValueType.INT16:   2,
    GGUFValueType.UINT32:  4,
    GGUFValueType.INT32:   4,
    GGUFValueType.FLOAT32: 4,
    GGUFValueType.UINT64:  8,
    GGUFValueType.INT64:   8,
    GGUFValueType.FLOAT64: 8,
    GGUFValueType.BOOL:    1
}

def unpack(value_type: GGUFValueType, file: BufferedReader):
    return struct.unpack(
        value_packing.get(value_type),
        file.read(value_lengths.get(value_type))
    )[0]

def get_single(
        value_type: GGUFValueType,
        file: BufferedReader
    ) -> Union[str, int, float, bool]:
    """Read a single value from an open file"""
    if value_type == GGUFValueType.STRING:
        string_length = unpack(GGUFValueType.UINT64, file=file)
        value = file.read(string_length)
        # officially, strings that cannot be decoded into utf-8 are invalid
        value = value.decode("utf-8")
    else:
        value = unpack(value_type, file=file)
    return value

def load_metadata(
        fn: Union[os.PathLike[str], str]
    ) -> dict[str, Union[str, int, float, bool, list]]:
    """
    Given a path to a GGUF file, peek at its header for metadata

    Return a dictionary where all keys are strings, and values can be
    strings, ints, floats, bools, or lists
    """

    metadata: dict[str, Union[str, int, float, bool, list]] = {}
    with open(fn, "rb") as file:
        magic = file.read(4)

        if magic != b"GGUF":
            raise ValueError(
                "your model file is not a valid GGUF file "
                f"(magic number mismatch, got {magic}, "
                "expected b'GGUF')"
            )
        
        version = unpack(GGUFValueType.UINT32, file=file)

        if version not in SUPPORTED_GGUF_VERSIONS:
            raise ValueError(
                f"your model file reports GGUF version {version}, but "
                f"only versions {SUPPORTED_GGUF_VERSIONS} "
                "are supported. re-convert your model or download a newer "
                "version"
            )
        
        tensor_count = unpack(GGUFValueType.UINT64, file=file)
        if version == 3:
            metadata_kv_count = unpack(GGUFValueType.UINT64, file=file)
        elif version == 2:
            metadata_kv_count = unpack(GGUFValueType.UINT32, file=file)

        for _ in range(metadata_kv_count):
            if version == 3:
                key_length = unpack(GGUFValueType.UINT64, file=file)
            elif version == 2:
                key_length = 0
                while key_length == 0:
                    # seek until next key is found
                    key_length = unpack(GGUFValueType.UINT32, file=file)
                file.read(4) # 4 byte offset for GGUFv2
            key = file.read(key_length)
            value_type = GGUFValueType(
                unpack(GGUFValueType.UINT32, file=file)
            )
            if value_type == GGUFValueType.ARRAY:
                array_value_type = GGUFValueType(
                    unpack(GGUFValueType.UINT32, file=file)
                )
                # array_length is the number of items in the array
                if version == 3:
                    array_length = unpack(GGUFValueType.UINT64, file=file)
                elif version == 2:
                    array_length = unpack(GGUFValueType.UINT32, file=file)
                    file.read(4) # 4 byte offset for GGUFv2
                array = [
                    get_single(
                        array_value_type,
                        file
                    ) for _ in range(array_length)
                ]
                metadata[key.decode()] = array
            else:
                value = get_single(
                    value_type,
                    file
                )
                metadata[key.decode()] = value

    return metadata
