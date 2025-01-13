class MemoryManager(object):

    def __init__(self) -> None:

        self.memory = {0: None, 1: None, 2: None}
        self.current_tracker = 0

    def get_trackers(self, position):
        bin_idx = self.get_bin_idx(position)
        if self.memory[bin_idx] is not None:
            return [value for (key, value) in self.memory[bin_idx].items()]
        else:
            return []

    def get_trackers_by_bin(self, bin_idx):
        if self.memory[bin_idx] is not None:
            return [value for (key, value) in self.memory[bin_idx].items()]
        else:
            return []

    def get_closest_tracker(self, position):
        min = 1000
        closest_tracker_list = None
        for memory_pose in self.memory.keys():
            if abs(memory_pose - position) < min:
                min = abs(memory_pose - position)
                closest_tracker_list = self.memory[memory_pose]
        for (_, value) in closest_tracker_list.items():
            if value.label is not None:
                return value

        return None

    def get_bin_idx(self, position):
        bin_idx = None
        if -100 <= position < -24:
            bin_idx = 0
        elif -24 <= position < 24:
            bin_idx = 1
        elif 24 <= position <= 100:
            bin_idx = 2
        return bin_idx

    def get_bin_idx_fromY(self, position):
        bin_idx = None
        if position:
            if position < -0.24:
                bin_idx = 0
            elif -0.24 <= position < -0.10:
                bin_idx = 1
            elif -0.10 <= position < 0.8:
                bin_idx = 2
            elif 0.8 <= position < 0.22:
                bin_idx = 3
            elif 0.22 <= position:
                bin_idx = 4
            return bin_idx
        else:
            return False

    def add_trackers(self, list_tracker, position):
        bin_idx = self.get_bin_idx(position)
        self.memory[bin_idx] = {t.id: t for t in list_tracker}

        if len(list_tracker) > self.current_tracker:
            self.current_tracker = len(list_tracker)

    def get_zone_by_label(self, t_label):
        positions = []
        print(self.memory)
        for position in self.memory.keys():
            print(self.memory.keys(), position)
            if self.memory[position] is not None:
                for tracker in self.memory[position]:
                    if self.memory[position][tracker].status == t_label:
                        positions.append(position)
        return positions

    def get_tracker_by_label(self, t_label):
        for position in self.memory.keys():
            for (_, tracker) in self.memory[position].items():
                if tracker.label == t_label and t_label != "unknown":
                    return tracker
        return None

    def get_tracker_by_id(self, t_id):
        for position in self.memory.keys():
            for (_, tracker) in self.memory[position].items():
                if tracker.id == t_id:
                    return tracker
        return None

