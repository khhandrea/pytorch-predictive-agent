from time import time

def seconds_to_hms(total_seconds: float) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f'{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}'

class ProgressFormatter:
    def __init__(self):
        self._is_first = True
        self._start_time = time()

    def print(self, data: dict[str, float]) -> None:
        # Check time
        elapsed = seconds_to_hms(time() - self._start_time)
        data = {'time elapsed': elapsed, **data}

        # Print once at first
        if self._is_first:
            print(' | '.join(content for content in data))
            self._is_first = False
        
        # Print after first time
        contents = []
        for content in data:
            if isinstance(data[content], float):
                contents.append(f'{data[content]:>{len(content)}.2f}')
            elif isinstance(data[content], int):
                contents.append(f'{data[content]:>{len(content)}d}')
            else:
                contents.append(f'{data[content]:>{len(content)}}')
        print(' | '.join(contents))