## runCustomInput(exec_name, input_template, param_dict, workdir='', label='runCustom')

Sometimes you want to run a program multiple times with slightly different
input files. When the input file is simple, it's straightforward to write a
script to create that file from scratch for each run. But if your input file is
large or complicated, there is a better option: to have a template file which
you update for each run with only the data that needs to change. 

This function does that. You tell is the executable name, the template file you
want to modify for each run, and a dictionary of parameters and the values you
want to set them to. Optionally, you can run the code within a directory which
is not your current working directory, and you can name the job. The function
then:

* Copies the template file, replacing keywords in the template that match keys 
in the dictionary with the values belonging to those keys
* Runs the executable with the custom input file just created 
