from collections import OrderedDict
from loguru import logger


class DialogLogger():
    def __init__(self, order, column_width=64, h_space=8, enabled=True):
        self.order = order
        self.column_width = column_width
        self.h_space = h_space
        # won't print anything if `self.enabled==False`
        self.enabled = enabled

    def _multi_column_log(self, print_func, **columns):
        for name in columns:
            if name not in self.order:
                raise ValueError(f"Unknown column: {name}")

        header_lens = OrderedDict(
            [(name, len(name) + 2) for name in self.order if name in columns]
        )
        first_line = True

        columns = {k: str(v) for k, v in columns.items()}

        while columns:
            # if there are still columns not completed
            line = ""

            for name, length in header_lens.items():
                # find a completed column -> print a blank line
                if name not in columns:
                    line += " " * (self.column_width + self.h_space)
                    continue

                # If `first_line`, add header, e.g. "Q: ", "A: ", "P1: "
                # If not `first_line`, add indentation to align with the first line
                header = f"{name}: " if first_line else (" " * length)

                # "\n" at the front -> print a blank line，possibly with header
                if columns[name].startswith("\n"):
                    columns[name] = columns[name][1:]
                    line += header.ljust(self.column_width + self.h_space)
                    continue

                columns[name] = header + columns[name]
                crop = columns[name][:self.column_width].split("\n")[0]
                columns[name] = columns[name][len(crop):]
                # remove the processed "\n"
                columns[name] = columns[name][columns[name].startswith("\n"):]

                line += crop.ljust(self.column_width + self.h_space)

                # remove the completed column
                if not columns[name]:
                    columns.pop(name)

            first_line = False
            print_func(line)
        print_func("-" * self.column_width)

    def info(self, **columns):
        if self.enabled:
            self._multi_column_log(logger.info, **columns)
