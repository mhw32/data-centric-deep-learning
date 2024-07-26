class sty:
  BLUE = "\033[0;34m"
  CYAN = "\033[0;36m"
  RED = "\033[0;31m"
  PINK = "\033[0;35m"
  GREEN = "\033[0;32m"
  BOLD = "\033[1m"
  ITALIC = "\033[3m"
  RESET = "\033[0m"


def color_print(color, *args, **kwargs):
  r"""Helper function for colored print
  """
  sep = kwargs.pop("sep", " ")
  end = kwargs.pop("end", "\n")
  text = sep.join(map(str, args))
  colored_text = "\n".join([color + line + sty.RESET for line in text.splitlines()])
  print(colored_text, end=end, **kwargs)


def pb(*args, **kwargs):
  color_print(sty.BLUE, *args, **kwargs)


def pc(*args, **kwargs):
  color_print(sty.CYAN, *args, **kwargs)


def pr(*args, **kwargs):
  color_print(sty.RED, *args, **kwargs)


def pp(*args, **kwargs):
  color_print(sty.PINK, *args, **kwargs)


def pg(*args, **kwargs):
  color_print(sty.GREEN, *args, **kwargs)
