import runner

# In this example, I use runCustom to run bigstick for several
# custom input files, where in each I change the number of 
# neutrons, and create a consitent input file.

exec_name = "bigstick.x"
input_template = "input.Z.N"
pnames = ["Znumber","Nnumber","2xJz","Anumber"]
workdir = "./runs/"
label="runner_example"

Z = 2
for N in range(2,4):
    A = N + Z
    if (A%2==0): # A is even
        Jzx2 = 0
    else:
        Jzx2 = 1
  
    pvalues = [Z, N, Jzx2, A]
    label = "isotope.%s.%s"%(Z,N)

    runner.runCustom(exec_name, input_template, pnames, pvalues, workdir, label)
