from plyfile import PlyData, PlyElement

def read_gt(path):
    plydata = PlyData.read(path)
    print(plydata)
    print("number of points = ", len(plydata.elements[0].data))

if __name__ == "__main__":
    object = "chair"
    read_gt(f"/home/hongyuzhou/Datasets/blend_files/{object}.ply")