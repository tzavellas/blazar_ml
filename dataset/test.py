from multiprocessing import Pool
from code_launcher import CodeLauncher


launcher = CodeLauncher(exec_path='/home/tzavellas/Documents/program.sh',
                        working_dir='/home/tzavellas/Downloads',
                        input='/home/tzavellas/Documents/out.csv')
# launcher = CodeLauncher(exec_path='/home/dino/PROJECTS_MP/code_clean/out',
#                         working_dir='/home/dino/PROJECTS_MP/test',
#                         input='/home/dino/PROJECTS_MP/hea_ML/hea_ml_project-master/out10.csv')

rows = launcher.get_inputs_dataframe().shape[0]

params = [0]
# for row in range(rows):
#    params.append(row)

with Pool() as pool:
    result = pool.map(launcher.run, params)

print(result)
