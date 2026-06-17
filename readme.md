

# Workshop Python Image analysis

This is a collection of qmd files that serve as the basis for an image analysis
workshop. 
The workshop is intended to be given in the Carpentries style.

This means the notebooks are teaching notes for *live coding* sessions, and 
contain exercises.
In the future, they are also intended as stand-alone course.

For more information, see the rendered version of this repository, at:

- https://jintram.github.io/Workshop_Py_image-analysis_2025/



# Convenient for instructors

### Creating pdfs

The following command can be used to generate pdfs (after navigating to the root dir of this
repository):

```sh
quarto render *.qmd --to pdf --output-dir pdf_py_ipynb --no-execute
```

### Creating .py and .ipynb files

To generate `.ipynb` files, use:

```sh
quarto render *.qmd --to ipynb --output-dir pdf_py_ipynb --no-execute
```

To generate python scripts, use:

```sh
jupyter nbconvert --to python "pdf_py_ipynb/*.ipynb" --output-dir pdf_py_ipynb
```

