import struct
from enum import IntEnum


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


_simple_value_packing = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
    GGUFValueType.BOOL: "?",
}

value_type_info = {
    GGUFValueType.UINT8: 1,
    GGUFValueType.INT8: 1,
    GGUFValueType.UINT16: 2,
    GGUFValueType.INT16: 2,
    GGUFValueType.UINT32: 4,
    GGUFValueType.INT32: 4,
    GGUFValueType.FLOAT32: 4,
    GGUFValueType.UINT64: 8,
    GGUFValueType.INT64: 8,
    GGUFValueType.FLOAT64: 8,
    GGUFValueType.BOOL: 1,
}


def get_single(value_type, file):
    if value_type == GGUFValueType.STRING:
        value_length = struct.unpack("<Q", file.read(8))[0]
        value = file.read(value_length)
        try:
            value = value.decode('utf-8')
        except:
            pass
    else:
        type_str = _simple_value_packing.get(value_type)
        bytes_length = value_type_info.get(value_type)
        value = struct.unpack(type_str, file.read(bytes_length))[0]

    return value


def load_metadata(fname):
    metadata = {}
    with open(fname, 'rb') as file:
        GGUF_MAGIC = struct.unpack("<I", file.read(4))[0]
        GGUF_VERSION = struct.unpack("<I", file.read(4))[0]
        ti_data_count = struct.unpack("<Q", file.read(8))[0]
        kv_data_count = struct.unpack("<Q", file.read(8))[0]
        
        if GGUF_VERSION == 1: 
            raise Exception('You are using an outdated GGUF, please download a new one.')

        for i in range(kv_data_count):
            key_length = struct.unpack("<Q", file.read(8))[0]
            key = file.read(key_length)

            value_type = GGUFValueType(struct.unpack("<I", file.read(4))[0])
            if value_type == GGUFValueType.ARRAY:
                ltype = GGUFValueType(struct.unpack("<I", file.read(4))[0])
                length = struct.unpack("<Q", file.read(8))[0]
                for j in range(length):
                    _ = get_single(ltype, file)
            else:
                value = get_single(value_type, file)
                metadata[key.decode()] = value

    return metadata
