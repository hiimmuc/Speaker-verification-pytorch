from __future__ import print_function
import sys
import grpc
import os
sys.path.append("/u01/os_call_monitor/CallEmotion/")
from asr import streaming_voice_pb2
from asr import streaming_voice_pb2_grpc
# from src.const import NORM_TEXT_API, ASR_SERVICES, CG_TEST_PATH
RATE = 48000
CHUNK_DURATION_MS = 30  # supports 10, 20 and 30 (ms)
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # chunk to read


def read_block(path_):
    with open(path_, 'rb') as rd:
        while True:
            block = rd.read(CHUNK_SIZE)   
            if len(block) == 0:
                break
            block_audio = streaming_voice_pb2.VoiceRequest(byte_buff=block)
            yield block_audio


def call_to_text(audio_path, channel):
    normalize_text = ""
    try:
        voice_bot_stub = streaming_voice_pb2_grpc.StreamVoiceStub(channel)
        res = voice_bot_stub.SendFile(read_block(audio_path))
        normalize_text = res.transcript
    except Exception as e:
        print("Error when calling ASR ::: " + str(e))
    return normalize_text


if __name__ == '__main__':

    data_dir = '../../../CallEmotion/data/cg_test/linhnvc/'
    calls = os.listdir(data_dir)
    # calls = os.listdir("/u01/os_call_monitor/CallEmotion/data/cg_test/linhnvc/")
    ports = ["8101", "8102", "8103", "8104", "8105", "8106", "8107", "8108"]
    for index_port, port in enumerate(ports):
        s_vice = '10.207.188.5:' + port
        channel_ = grpc.insecure_channel(s_vice)
        text = call_to_text(data_dir+calls[0], channel_)
        print(s_vice, '>>>>>>>>', text, '\n')


