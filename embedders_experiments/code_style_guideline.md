# General Code Instructions
plit your code into functions. Write functions that
each solve a single well-defined task 
are documented 
are testable; bonus points for unit tests
ideally, the output is via the return value only. If any parameters have to be modified, document it
Use self-explanatory variable and function names
Document all functions (what they do, what is input, what is output); add other comments as appropriate 
You must use appropriate libraries to handle JSON, XML, csv, etc. Do not write the code yourself. 

Strictly follow the PEP8 code style; a reasonable IDE (e.g., IntelliJ IDEA, PyCharm, MS VS Code) checks this for you.
Use type hints for function parameters and return values. First, type hints make it easier to understand your code. Second, they help you do avoid many errors (a reasonable IDE or a type checker, e.g., http://www.mypy-lang.org/ will do the code analysis). Use them at least for basic types (str, int, List, Set, Iterable, Tuple, Optional, Mapping, Dict, ...). You do not need to use them for complex types of third-party libraries where it is not clear what the type is. 
Include a pyproject.toml, requirements.txt, or setup.py file so I can easily install third-party libraries. Make sure that all libraries have their versions specified.
Use the main function pattern. 
Use argparse to parse arguments,  pathlib's Path for all files and directories, Counter to count objects. 
Use with to manipulate files.
Do not put executable code directly into a file (at the module's top level). It gets executed the moment you import it into another file. This is usually an unwanted and unexpected side effect. 