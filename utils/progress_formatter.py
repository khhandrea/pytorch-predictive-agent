class ProgressFormatter:
    def __init__(self):
        self._is_first = True

    def print(self, data: dict[str, float]) -> None:
        if self._is_first:
            print(' | '.join(content for content in data))
            self._is_first = False
        
        contents = []
        for content in data:
            if isinstance(data[content], float):
                contents.append(f'{data[content]:>{len(content)}.2f}')
            elif isinstance(data[content], int):
                contents.append(f'{data[content]:>{len(content)}d}')
            else:
                contents.append(f'{data[content]:>{len(content)}}')
        print(' | '.join(contents))