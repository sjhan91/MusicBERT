class Item:
    def __init__(self, name, start, end, velocity=None, pitch=None, instrument=None):
        self.name = name
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.instrument = instrument

    def __repr__(self):
        return "Item(name={}, start={}, end={}, pitch={}, velocity={}, instrument={})".format(
            self.name, self.start, self.end, self.pitch, self.velocity, self.instrument
        )


class Event:
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return "Event(name={}, time={}, value={}, text={})".format(
            self.name, self.time, self.value, self.text
        )
