import numpy as np


class ReadAndProcessOnTheFly:
    """Read from an open fileobject on the fly and do some processing on
    new data that is written to it. Files should be opened using a 'with open'
    statement to be sure that they are closed.

    To do
    use with open in here. Point at current pos and read N finished blocks. Put
    pointer at that position and return traj. If only some frames ready, point
    at last whole ready block read and return [] or the ready frames.
    """

    def __init__(self, file_path, processing_function, read_mode="r"):
        self.file_path = file_path
        self.processing_function = processing_function
        self.current_position = 0
        self.file_object = None
        self.read_mode = read_mode

    def read_and_process_content(self):
        # we may open at a time where the file
        # is currently not open for reading
        try:
            with open(self.file_path, self.read_mode) as self.file_object:
                self.file_object.seek(self.current_position)
                self.previous_position = self.current_position
                trajectory = self.processing_function(self)
                return trajectory
        except FileNotFoundError:
            return []


def lammpstrj_processer(reader_class):
    """
    Assumes the foolowing format
        dump custom id x y z vx vy vz ..rest ignored
        dump modify sorted
    giving the following structure

    ITEM: TIMESTEP
    t
    ITEM: NUMBER OF ATOMS
    N_atoms
    ITEM: BOX BOUNDS pp pp pp
    0 xhi
    0 yhi
    0 zhi
    ITEM: ATOMS id x y z vx vy vz
    ...
    sorted atoms
    ...
    """
    # trajectory of ready frames to be returned
    trajectory = []
    # holder for storing frame coordinates
    frame_coordinates = []
    for i, line in enumerate(reader_class.file_object.readlines()):
        spl = line.split()
        if i == 3 and spl:
            N_atoms = int(spl[0])
            block_size = N_atoms + 2  # 2 header lines
        # if we are not in the atom nr or header block
        if i % block_size > 1:
            # if there arent enough values to iterate through
            # return the (posibly empty) ready trajectory frames
            if len(spl) != 4:
                reader_class.current_position = reader_class.previous_position
                return trajectory
            else:
                frame_coordinates.append([float(spl[i]) for i in range(1, 4)])
        # if we are done with one block
        # update the file object pointer to the new position
        if i % block_size == N_atoms + 1 and i > 0:
            trajectory.append(np.array(frame_coordinates))
            reader_class.current_position = reader_class.file_object.tell()
            frame_coordinates = []

    return trajectory
