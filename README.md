# Smartautomation Benchmark
This benchmark provides a total of_______ python files, categorized depending on the size of the file, ... . This benchmark can be used to assess the quality of a commenting tool for python files.


## Benchmark evaluation
The problem for assessing the quality of comments is, that not all aspects of a comment can be classified as correct or incorrect by a set of rules, for example the meaning of the comment. Maybe one developer thinks that his comment describes a function perfectly, but another developer thinks that the comment misses to explain important functionalities of the code. For this reason this benchmark evaluation strategy focuses on qualities of a comment that can be measured, for example if it adheres to the PEP 8 Python guidelines.

What to look out for in a comment:
- Does every comment adhere to the PEP 8 guideline?
- Does every module, function, class and method have a docstring as its first statement?
- Does every docstring adhere to the PEP 257 guideline?
- Does every descriptive comment have a maximum length of 2 sentences?
- Is there NOT an excessive use of inline comments?
- Is the commenting style consistent for the whole file?
- Does the automatic comment generation NOT disrupt any code?
- Is there only one comment for a logical code section?
- Is every comment written in enlish, has no grammar mistakes and is understandable?
-  


## What are good comments?
Wether a comment is "good" for a particular code segment can be entirely subjective, however, there are some best practices defined over the years by the programming- and python community. Sticking to these best practices ensures a certain quality of each comment and 


### comment types
- **Block comment**: spans across multiple lines.
- **Line comment**: spans only one line.
- **Prologue comment**: located near the top of an associated programming topic, such as a symbol declaration or at the top of a file.
- **Inline comment**: located on the same line and to the right of programm code to which it refers.
  - Best practices: Follow standard formatting conventions. Avoid redundancy by using code alone or in combination.
- **Docstrings**: The docstring for a Python code object (a module, class, or function) is the first statement of that code object, immediately following the definition (the 'def' or 'class' statement, or at the top of the file for module). The statement must be a bare string literal, not any other kind of expression.
  - Best practices: Use single line docstrings for simple functions or methods, and multi line docstrings for complex classes or functions.


### Use-cases
- **Describe intent**: Explains the authors intent - why the code is as it is.
- **Highlight unusual practices**: Explains why a choice was made to write code that is counter to convention or best practices.
- **Describe algorithm**: Similar to pseudocode.
- **Reference**: When some aspect of the code is based on information in an external reference, comments link to the reference.
- **Comment out**: Part of the code that is not needed right now but might become relevant again in the future.
- **Store metadata**: Common metadata includes the name of the original author and subsequent maintainers, dates when first written and modified, link to development and user documentation, and legal information such as copyright and software license.
- **Integrate with development tools**: Sometimes information stored in comments is used by development tools other than the translator – the primary tool that consumes the code. This information may include metadata (often used by a documentation generator) or tool configuration.
- **Support documentation generation**: An API documentation generator parses information from a codebase to generate API documentation. Many support reading information from comments, often parsing metadata, to control the content and formatting of the resulting document.
- **Visualization**: An ASCII art visualization such as a logo, diagram, or flowchart can be included in a comment.
- **Store resource data**: *Not relevant for us*
- **Document development process**: Sometimes, comments describe development processes related to the code. For example, comments might describe how to build the code or how to submit changes to the software maintainer.


### Need for comments and level of detail
There is no rule how well source code should be documented, some developers think that there should as less comments as possible and the code should be self-documenting. Others suggest code should be extensively commented, it s not uncommon for over 50% of the non-whitespace characters in source code to be contained within comments. While more comments don't hurt, the programmer should pay attention to the level of detail in their comments. Stating the obvious in comments doesn't help most of the times and only litters the code unnecessarily.


### Styles
While every developer is free to style their comments however they want, a consitent, non-obstructive, easy to modify and difficult to break style is preferred. Python developers often categorize comments into high-level or low-level (detailed). High-level usually relates to the 'why' behind the code, Low-level is concerned with the 'how', specifically detailing any steps or thought process that went into reaching that decision.
Best Practice to Write Comments:
1. Comments should be short and precise.
2. Use comments only when necessary, don't clutter your code comments.
3. Avoid writing generic or basic comments.
4. Write comments that are self explanatory.


## Best practices for Python code
- **Be concise**: Keep your comments brief and to the point. A good comment should take no more than a sentence or two to read.
- **Use clear language**: Avoid jargon or obscure terminology that might confuse others (or yourself)
- **Avoid excessive commenting**: Don't comment every single line of code. This can be overwhelming and diminish the impact of your comments.
- **Focus on understanding**: A good comment often reveals the thought process behind a piece of code.
- **Keep comments up-to-date**: Ensure that your comments accurately reflect the current state of the code. Update comments whenever you modify the code to maintain consistency and avoid confusion.
- **Explain why, not just what**: Focus on explaining the reasoning behind the code rather than merely describing what the code does. Provide insights into the decision-making process and the problem being solved.
- **Use proper grammar and spelling**: Treat comments as a form of documentation. Use proper grammar, spelling, and punctuation to enhance readability and professionalism.
- **Don't state the obvious**: Avoid writing comments that merely restate what the code already clearly expresses. Comments should add value and provide additional insights.
- **Balance comment quantity**: Strike a balance between under-commenting and over-commenting. Too many comments can clutter the code and make it harder to read. Aim for self-explanatory code whenever possible.
- **Place Comments close to relevant code**: Position comments near the code they describe. This proximity makes it easier for readers to understand the context and purpose of the comments.
- **Use consistent style**: Follow a consistent style for writing comments throughout your codebase. Consistency improves readability and maintainability.


### Commenting standards and style guides
1. **PEP 8**: PEP 8 is the official style guide for Python code. It provides recommendations for comment formatting, indentation, and line length.
2. **Google python style guide**: Google‘s Python style guide offers guidelines for writing comments, including the use of docstrings and inline comments.
3. **NumPy/SciPy Docstring Guide**: The NumPy and SciPy projects have their own docstring conventions that are widely used in the scientific Python community.

There are tools that auto-generate documentation based on docstrings of python code, so it would be helpful if the comment is structured in a way, that these automation programs can use the comments, if a developer wants to use them.

### Relevant PEP 8 style guidelines
- **Maximum line length**: Limit all lines to a maximum of 79 characters. For flowing long blocks of text with fewer strucural restrictions (docstrings or comments), the line length should be limited to 72 characters.
- **Blank lines**:
  - Surround top-level function and class definitions with two blank lines.
  - Method definitions inside a class are surrounded by a single blank line.
  - Use blank lines in functions, sparingly, to indicate logical sections.
- **Comments**:
  - Should be complete sentences.The first word should be capitalized, unless it is an identifier that begins with a lower case letter (never alter the case of identifiers!).
  - Block comments generally consist of one or more paragraphs built out of complete sentences, with each sentence ending in a period.
  - You should use one or two spaces after a sentence-ending period in multi-sentence comments, except after the final sentence.
  - Ensure that your comments are clear and easily understandable to other speakers of the language you are writing in.
  - Python coders from non-English speaking countries: please write your comments in English, unless you are 120% sure that the code will never be read by people who don’t speak your language.
  - **Block comments**: Each line of a block comment starts with a # and a single space (unless it is indented text inside the comment). Paragraphs inside a block comment are separated by a line containing a single #.
  - **Inline comments**: Use inline comments sparingly. An inline comment is a comment on the same line as a statement. Inline comments should be separated by at least two spaces from the statement. They should start with a # and a single space. Inline comments are unnecessary and in fact distracting if they state the obvious.
  - **Docstrings**: Write docstrings for all public modules, functions, classes, and methods. Docstrings are not necessary for non-public methods, but you should have a comment that describes what the method does. This comment should appear after the def line. For more see PEP 257.

### PEP 257 style guidelines for docstrings
A docstring is a string literal that occurs as the first statement in a module, function, class, or method definition. In contrast to block or inline comments, docstrings are recognized as python objects by the compiler.

- **One-line docstrings**: 
  - The closing quotes are on the same line as the opening quotes.
  - There’s no blank line either before or after the docstring.
  - The docstring is a phrase ending in a period. It prescribes the function or method’s effect as a command (“Do this”, “Return that”), not as a description; e.g. don’t write “Returns the pathname …”.
  - The one-line docstring should NOT be a “signature” reiterating the function/method parameters.
- **Multi-line docstings**:
  - Multi-line docstrings consist of a summary line just like a one-line docstring, followed by a blank line, followed by a more elaborate description. The summary line may be used by automatic indexing tools; it is important that it fits on one line and is separated from the rest of the docstring by a blank line. The summary line may be on the same line as the opening quotes or on the next line. The entire docstring is indented the same as the quotes at its first line.
  - Insert a blank line after all docstrings (one-line or multi-line) that document a class.
  - The docstring for a module should generally list the classes, exceptions and functions (and any other objects) that are exported by the module, with a one-line summary of each.
  - The docstring for a package should also list the modules and subpackages exported by the package.
  - The docstring for a function or method should summarize its behavior and document its arguments, return value(s), side effects, exceptions raised, and restrictions on when it can be called. Optional arguments should be indicated. It should be documented whether keyword arguments are part of the interface.
  - The docstring for a class should summarize its behavior and list the public methods and instance variables. If the class is intended to be subclassed, and has an additional interface for subclasses, this interface should be listed separately (in the docstring). The class constructor should be documented in the docstring for its __init__ method. Individual methods should be documented by their own docstring.
  - If a class subclasses another class and its behavior is mostly inherited from that class, its docstring should mention this and summarize the differences. Use the verb “override” to indicate that a subclass method replaces a superclass method and does not call the superclass method; use the verb “extend” to indicate that a subclass method calls the superclass method (in addition to its own behavior).


## Sources
https://en.wikipedia.org/wiki/Comment_%28computer_programming%29
https://en.wikipedia.org/wiki/Docstring
https://www.geeksforgeeks.org/python-comments/
https://www.machinelearnguru.com/how-to-use-comments-effectively-in-python
https://www.33rdsquare.com/comments-in-python/
https://peps.python.org/pep-0008/
https://peps.python.org/pep-0257/