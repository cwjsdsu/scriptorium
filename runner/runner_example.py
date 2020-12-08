import runCustomInput

"""
In this example, I use runCustom to run bigstick for several
custom input files, where in each I change the number of 
neutrons, and create a consistent input file.

"""

exec_name = "bigstick.x"
input_template = "input.Z.N"
workdir = "./runs/"
label="runner_example"

core=16
Z = 2
for N in range(2,4):
    A = N + Z + core
    if (A%2==0): # A is even
        Jzx2 = 0
    else:
        Jzx2 = 1
    parameters = { 'Znumber' : Z,
                   'Nnumber' : N,
                   '2xJz' : Jzx2,
                   'Anumber': A }
    label = "isotope.%s.%s"%(Z,N)

    #runCustomInput.runCustom(exec_name, input_template, pnames, pvalues, workdir, label)
    runCustomInput.runCustomInput(exec_name, input_template, parameters, workdir, label)
