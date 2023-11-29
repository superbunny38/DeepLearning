print("output file directory exist:",os.path.isdir(output_dir))
if os.path.isdir(output_dir) == True:
    try:
        assert len(os.listdir(output_dir)) == 0, print(f"output folder not empty! # files: {len(os.listdir(output_dir))}")
    except:
        if len(os.listdir(output_dir)) != 0:
            shutil.rmtree(output_dir)
        assert len(os.listdir(output_dir)) == 0
else:
    os.mkdir(output_dir)

print("#files in output folder:",len(os.listdir(output_dir)))