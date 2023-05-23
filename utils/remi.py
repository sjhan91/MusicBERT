import numpy as np
import pretty_midi

from utils.constants import *
from utils.remi_utils import *
from utils.chord_recognition import MIDIChord

"""
Refer to https://github.com/dvruette/figaro/blob/2b253eb0476453e197ee0599a5c58f87d82a3890/src/input_representation.py
"""

#####################################################################
# CONVERT TO REMI
#####################################################################
class REMI:
    def __init__(self, file_path):
        self.pm = pretty_midi.PrettyMIDI(file_path)

        if not self.check_time_sign(self.pm, num=4, denom=4):
            raise ValueError("Invalid MIDI file: invalid time signature")

        self.resolution = self.pm.resolution
        self.ticks = self.resolution / DEFAULT_POS_PER_QUARTER

        self.read_items()
        self.quantize_items()
        self.extract_chords()
        self.group_items()

        if len(self.note_items) == 0:
            raise ValueError("Invalid MIDI file: no notes found, empty file")

    def check_time_sign(self, pm, num=4, denom=4):
        time_sign_list = pm.time_signature_changes

        # empty check
        if len(time_sign_list) == 0:
            return False

        # nom and denom check
        for time_sign in time_sign_list:
            if time_sign.numerator != num or time_sign.denominator != denom:
                return False

        return True

    def read_items(self):
        ##### Note
        self.note_items = []
        for inst in self.pm.instruments:
            start = None
            pedal_pressed = False
            pedal_events = [event for event in inst.control_changes if event.number == 64]

            pedals = []
            for e in pedal_events:
                if e.value >= 64 and not pedal_pressed:
                    pedal_pressed = True
                    start = e.time
                elif e.value < 64 and pedal_pressed:
                    pedal_pressed = False
                    pedals.append(Item(name="Pedal", start=start, end=e.time))

            notes = inst.notes
            notes.sort(key=lambda x: (x.start, x.pitch))

            if inst.is_drum:
                inst_name = "Drum"
            else:
                inst_name = int(inst.program)

            pedal_idx = 0
            for note in notes:
                pedal_candidates = [
                    (i + pedal_idx, pedal)
                    for i, pedal in enumerate(pedals[pedal_idx:])
                    if note.end >= pedal.start and note.start < pedal.end
                ]

                if len(pedal_candidates) > 0:
                    pedal_idx = pedal_candidates[0][0]
                    pedal = pedal_candidates[-1][1]
                else:
                    pedal = Item(name="Pedal", start=0, end=0)

                self.note_items.append(
                    Item(
                        name="Note",
                        start=self.pm.time_to_tick(note.start),
                        end=self.pm.time_to_tick(max(note.end, pedal.end)),
                        velocity=note.velocity,
                        pitch=note.pitch,
                        instrument=inst_name,
                    )
                )

        self.note_items.sort(key=lambda x: (x.start, x.pitch))

        ##### Tempo
        self.tempo_items = []
        times, tempi = self.pm.get_tempo_changes()

        for time, tempo in zip(times, tempi):
            self.tempo_items.append(
                Item(
                    name="Tempo",
                    start=self.pm.time_to_tick(time),
                    end=None,
                    velocity=None,
                    pitch=int(tempo),
                )
            )

        self.tempo_items.sort(key=lambda x: x.start)

    def quantize_items(self):
        # grid
        end_tick = self.pm.time_to_tick(self.pm.get_end_time())
        grids = np.arange(0, max(self.resolution, end_tick), self.ticks)

        # process
        for item in self.note_items:
            index = np.searchsorted(grids, item.start, side="right")

            if index > 0:
                index -= 1

            shift = round(grids[index]) - item.start

            item.start += shift
            item.end += shift

    def group_items(self):
        items = self.chords + self.tempo_items + self.note_items

        def get_key(item):
            type_priority = {"Chord": 0, "Tempo": 1, "Note": 2}

            return (
                item.start,
                type_priority[item.name],
                -1 if item.instrument == "Drum" else item.instrument,
                item.pitch,
            )

        items.sort(key=get_key)

        downbeats = self.pm.get_downbeats()
        downbeats = np.concatenate([downbeats, [self.pm.get_end_time()]])

        self.groups = []
        for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
            db1, db2 = self.pm.time_to_tick(db1), self.pm.time_to_tick(db2)

            insiders = []
            for item in items:
                if (item.start >= db1) and (item.start < db2):
                    insiders.append(item)

            overall = [db1] + insiders + [db2]
            self.groups.append(overall)

        # trim empty groups from the begin and end
        for idx in [0, -1]:
            while len(self.groups) > 0:
                group = self.groups[idx]
                notes = [item for item in group[1:-1] if item.name == "Note"]

                if len(notes) == 0:
                    self.groups.pop(idx)
                else:
                    break

        return self.groups

    def tick_to_pos(self, tick):
        return round(tick / self.resolution * DEFAULT_POS_PER_QUARTER)

    # extract chord
    def extract_chords(self):
        end_tick = self.pm.time_to_tick(self.pm.get_end_time())

        if end_tick < self.resolution:
            self.chords = []
            return self.chords

        method = MIDIChord(self.pm)
        chords = method.extract()

        output = []
        for chord in chords:
            output.append(
                Item(
                    name="Chord",
                    start=self.pm.time_to_tick(chord[0]),
                    end=self.pm.time_to_tick(chord[1]),
                    velocity=None,
                    pitch=chord[2].split("/")[0],
                )
            )

        if len(output) == 0 or output[0].start > 0:
            if len(output) == 0:
                end = self.pm.time_to_tick(self.pm.get_end_time())
            else:
                end = output[0].start

            output.append(Item(name="Chord", start=0, end=end, velocity=None, pitch="N:N"))

        self.chords = output

    # item to event
    def get_remi_events(self):
        tempo_curr = self.tempo_items[0]
        tempo_idx = np.argmin(abs(DEFAULT_TEMPO_BINS - tempo_curr.pitch))

        meta_info = {}
        meta_info["inst"] = []
        meta_info["chord"] = []
        meta_info["tempo"] = []
        meta_info["mean_velocity"] = []
        meta_info["mean_duration"] = []
        meta_info["groove_pattern"] = []

        events = []
        n_downbeat = 0
        for i, group in enumerate(self.groups):
            bar_st, bar_et = group[0], group[-1]
            n_downbeat += 1

            # <Bar>
            events.append(
                Event(
                    name=BAR_KEY,
                    time=None,
                    value="{}".format(0),
                    text="{}".format(n_downbeat - 1),
                )
            )

            # <Tempo>
            events.append(
                Event(
                    name=POSITION_KEY,
                    time=0,
                    value="{}".format(0),
                    text="{}/{}".format(1, DEFAULT_POS_PER_BAR),
                )
            )

            tempos = [item for item in group[1:-1] if item.name == "Tempo"]

            if len(tempos) > 0:
                tempo_curr = tempos[0]
                tempo_idx = np.argmin(abs(DEFAULT_TEMPO_BINS - tempo_curr.pitch))

            events.append(
                Event(
                    name=TEMPO_KEY,
                    time=tempo_curr.start,
                    value=tempo_idx,
                    text="{}/{}".format(tempo_curr, DEFAULT_TEMPO_BINS[tempo_idx]),
                )
            )

            ticks_per_bar = self.resolution * DEFAULT_QUARTERS_PER_BAR
            flags = np.linspace(bar_st, bar_st + ticks_per_bar, DEFAULT_POS_PER_BAR, endpoint=False)

            temp_inst = []
            temp_chord = []
            temp_velocity = []
            temp_duration = []
            temp_groove_pattern = []
            for item in group[1:-1]:
                # <Pos>
                pos_idx = np.argmin(abs(flags - item.start))

                pos_event = Event(
                    name=POSITION_KEY,
                    time=item.start,
                    value="{}".format(pos_idx),
                    text="{}/{}".format(pos_idx + 1, DEFAULT_POS_PER_BAR),
                )

                if item.name == "Note":
                    events.append(pos_event)

                    if item.instrument == "Drum":
                        name = "Drum"
                    else:
                        name = pretty_midi.program_to_instrument_name(item.instrument)

                    # <Inst>
                    events.append(
                        Event(
                            name=INSTRUMENT_KEY,
                            time=item.start,
                            value=name,
                            text="{}".format(name),
                        )
                    )

                    # <Pitch>
                    events.append(
                        Event(
                            name=PITCH_KEY,
                            time=item.start,
                            value="Drum_{}".format(item.pitch) if name == "Drum" else item.pitch,
                            text="{}".format(pretty_midi.note_number_to_name(item.pitch)),
                        )
                    )

                    # <Velocity>
                    velocity_idx = np.argmin(abs(DEFAULT_VELOCITY_BINS - item.velocity))
                    events.append(
                        Event(
                            name=VELOCITY_KEY,
                            time=item.start,
                            value=velocity_idx,
                            text="{}/{}".format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_idx]),
                        )
                    )

                    # <Duration>
                    duration = self.tick_to_pos(item.end - item.start)
                    duration_idx = np.argmin(abs(DEFAULT_DURATION_BINS - duration))
                    events.append(
                        Event(
                            name=DURATION_KEY,
                            time=item.start,
                            value=duration_idx,
                            text="{}".format(duration_idx),
                        )
                    )

                    temp_inst.append(128 if item.instrument == "Drum" else item.instrument)
                    temp_velocity.append(item.velocity)
                    temp_duration.append(duration_idx)
                    temp_groove_pattern.append(pos_idx)

                elif item.name == "Chord":
                    chord, tone = item.pitch.split(":")

                    # remove "N" or None
                    if not ((chord == "N" or chord == "None") or (tone == "N" or tone == "None")):
                        temp_chord.append(item.pitch)

            meta_info["inst"].append(list(set(temp_inst)))
            meta_info["chord"].append(temp_chord)
            meta_info["tempo"].append(DEFAULT_TEMPO_BINS[tempo_idx])
            meta_info["mean_velocity"].append(np.mean(temp_velocity))
            meta_info["mean_duration"].append(np.mean(temp_duration))
            meta_info["groove_pattern"].append(list(set(temp_groove_pattern)))

        return [f"{e.name}_{e.value}" for e in events], meta_info


#####################################################################
# WRITE MIDI
#####################################################################
def remi2midi(events, bpm=120, time_signature=(4, 4)):
    def get_time(reference, bar, pos):
        ref_pos = reference["pos"]

        d_bar = bar - ref_pos[0]
        d_pos = (pos - ref_pos[1]) + (d_bar * DEFAULT_POS_PER_BAR)
        d_quart = d_pos / DEFAULT_POS_PER_QUARTER
        dt = d_quart / reference["tempo"] * 60

        return reference["time"] + dt

    num, denom = time_signature
    tempo_changes = [event for event in events if f"{TEMPO_KEY}_" in event]

    if len(tempo_changes) > 0:
        bpm = DEFAULT_TEMPO_BINS[int(tempo_changes[0].split("_")[-1])]

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=480)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(num, denom, 0))

    last_tl_event = {"time": 0, "pos": (0, 0), "tempo": bpm}

    bar = -1
    instruments = {}
    for i, event in enumerate(events):
        # bar event
        if f"{BAR_KEY}_" in event:
            bar += 1

        # tempo event
        elif (
            i + 1 < len(events)
            and f"{POSITION_KEY}_" in events[i]
            and f"{TEMPO_KEY}_" in events[i + 1]
        ):
            pos = int(events[i].split("_")[-1])
            tempo_idx = int(events[i + 1].split("_")[-1])
            tempo = DEFAULT_TEMPO_BINS[tempo_idx]

            if tempo != last_tl_event["tempo"]:
                time = get_time(last_tl_event, bar, pos)

                last_tl_event["time"] = time
                last_tl_event["pos"] = (bar, pos)
                last_tl_event["tempo"] = tempo

        # note event
        elif (
            i + 4 < len(events)
            and f"{POSITION_KEY}_" in events[i]
            and f"{INSTRUMENT_KEY}_" in events[i + 1]
            and f"{PITCH_KEY}_" in events[i + 2]
            and f"{VELOCITY_KEY}_" in events[i + 3]
            and f"{DURATION_KEY}_" in events[i + 4]
        ):
            # get position
            pos = int(events[i].split("_")[-1])

            # get instrument
            inst_name = events[i + 1].split("_")[-1]

            if inst_name not in instruments:
                if inst_name == "Drum":
                    inst = pretty_midi.Instrument(0, is_drum=True)
                else:
                    program = pretty_midi.instrument_name_to_program(inst_name)
                    inst = pretty_midi.Instrument(program)

                instruments[inst_name] = inst
            else:
                inst = instruments[inst_name]

            # get pitch
            pitch = int(events[i + 2].split("_")[-1])

            # get velocity
            velocity_idx = int(events[i + 3].split("_")[-1])
            velocity = DEFAULT_VELOCITY_BINS[velocity_idx]

            # get duration
            duration = int(events[i + 4].split("_")[-1])

            # create pm object
            start = get_time(last_tl_event, bar, pos)
            end = get_time(last_tl_event, bar, pos + duration)

            note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
            inst.notes.append(note)

    for instrument in instruments.values():
        pm.instruments.append(instrument)

    return pm
