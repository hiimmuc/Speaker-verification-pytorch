import collections
import contextlib
import glob
import os
import sys
import time
import wave
import numpy as np

import webrtcvad
 
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# VAD utilities


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, path
        sample_width = wf.getsampwidth()
        assert sample_width == 2, path
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000), path
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class VAD:
    def __init__(self,  mode=3, frame_duration=30, win_length=300) -> None:
        self.mode = mode
        self.frame_duration = frame_duration
        self.win_length = win_length
        self.vad = webrtcvad.Vad(int(mode))

    def frame_generator(self, audio, sample_rate):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        frame_duration_ms = self.frame_duration
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frames, show=False):
        """Filters out non-voiced audio frames.
        Given a webrtcvad.Vad and a source of audio frames, yields only
        the voiced audio.
        Uses a padded, sliding window algorithm over the audio frames.
        When more than 90% of the frames in the window are voiced (as
        reported by the VAD), the collector triggers and begins yielding
        audio frames. Then the collector waits until 90% of the frames in
        the window are unvoiced to detrigger.
        The window is padded at the front and back to provide a small
        amount of silence or the beginnings/endings of speech around the
        voiced frames.
        Arguments:
        sample_rate - The audio sample rate, in Hz.
        frame_duration_ms - The frame duration in milliseconds.
        padding_duration_ms - The amount to pad the window, in milliseconds.
        vad - An instance of webrtcvad.Vad.
        frames - a source of audio frames (sequence or generator).
        Returns: A generator that yields PCM audio data.
        """
        padding_duration_ms = self.win_length
        frame_duration_ms = self.frame_duration
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        triggered = False

        voiced_frames = []
        unvoiced_frames = []
        
        for frame in frames:
            is_speech = self.vad.is_speech(frame.bytes, sample_rate)
            if show:
                sys.stdout.write('1' if is_speech else '_')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    if show:
                        sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
                    unvoiced_frames = []

            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    if show:
                        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                        for f, s in ring_buffer:
                            unvoiced_frames.append(f)

                    triggered = False
                    
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
                    
        if triggered:
            if show:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        if show:
            sys.stdout.write('\n')
        # If we have any leftover voiced audio when we run out of input,
        # yield it.
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames])

    def detect(self, audio_path, write=True, overwrite=False, show=False):
        if not os.path.exists(audio_path):
            raise "Path is not existed"
        audio, sample_rate = read_wave(audio_path)

        frames = self.frame_generator(audio, sample_rate)
        frames = list(frames)

        segments = self.vad_collector(sample_rate, frames, show=show)

        if write:
            for i, segment in enumerate(segments):
                if len(segment) / sample_rate >= 1.0:
                    path = f"{audio_path.replace('.wav', '')}_vad_{i}.wav"
                    write_wave(path, segment, sample_rate)

        segments = [np.frombuffer(seg) for seg in segments]
        return segments

