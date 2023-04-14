from infretis.classes.fileio import OutputBase

class ScreenOutput(OutputBase):
    """A class for handling output to the screen."""

    target = 'screen'

    def write(self, towrite, end=None):
        """Write a string to the file.

        Parameters
        ----------
        towrite : string
            The string to output to the file.
        end : string, optional
            Override how the print statements ends.

        Returns
        -------
        status : boolean
            True if we managed to write, False otherwise.

        """
        if towrite is None:
            return False
        if end is not None:
            print(towrite, end=end)
            return True
        print(towrite)
        return True


def print_to_screen(txt=None, level=None):  # pragma: no cover
    """Print output to standard out.

    This method is included to ensure that output from PyRETIS to the
    screen is written out in a uniform way across the library and
    application(s).

    Parameters
    ----------
    txt : string, optional
        The text to write to the screen.
    level : string, optional
        The level can be used to color the output.

    """
    if txt is None:
        print()
    else:
        out = '{}'.format(txt)
        color = _PRINT_COLORS.get(level, None)
        if color is None:
            print(out)
        else:
            print(color + out)
