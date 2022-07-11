#import ujson as json
import pandas as pd

def grep(tracks, p, s, e):
    primary = []
    neighbours = []
    count = 0
    while count < 100:
        for track in tracks:
            if track['f'] >= s and track['f'] <= e:
                if track['p'] == p:
                    primary.append(track)
                    count += 1
                else:
                    neighbours.append(track)
                    count += 1
    return primary, neighbours

def create_scene(tracks, s_id, p, s, e, tag):
    scene_list = []
    scene = []
    scene.append(s_id)
    primary, neighbours = grep(tracks, p, s, e)
    # {"f": 95, "p": 19, "x": 0.21, "y": 0.45}
    
    for track in sorted(primary, key=lambda x: (x['p'], x['f'])):
        scene.extend([track['x'], track['y']])

    for track in sorted(neighbours, key=lambda x: (x['p'], x['f'])):
        scene.extend([track['x'], track['y']])

    for i in range(len(tag[-1])):
        sc_copy = scene
        sc_copy.append(tag[-1][i])
        scene_list.append(sc_copy)

    return scene_list

def create_categories():
    categories = ['ID']
    pedestrians = ['P', 'N1', 'N2', 'N3', 'N4']
    for pedestrian in pedestrians:
        for i in range(21):
            x = pedestrian + 'X' + str(i)
            y = pedestrian + 'Y' + str(i)
            categories.extend([x, y])
    categories.append('TAG')

    return categories

if __name__ == "__main__":
    train_path = "five_parallel_synth/train/orca_five_nontraj_synth.ndjson"
    df = pd.read_json(train_path, lines=True)

    # hard coded for efficiency
    df_scenes = df.iloc[0:43697, 0]
    df_tracks = df.iloc[43698: , 1]  
    
    categories = create_categories()
    with open('five_parallel_synth.dat', 'w') as f:
        f.write('\t'.join([x for x in categories]))

    print("Categories written.")
    scenes = []

    for df_scene in df_scenes:
        s_id = df_scene['id']
        p = df_scene['p']
        s = df_scene['s']
        e = df_scene['e']
        tag = df_scene['tag']

        scene_list = create_scene(df_tracks, s_id, p, s, e, tag)
        print("Creating scene id:", s_id)
        for scene in scene_list:
            scene_str = '\t'.join([str(x) for x in scene])
            scenes.append(scene_str)

    with open('five_parallel_synth.dat', 'a') as f:
        f.write('\n')
        for scene in scenes:
            f.write(f"{scene}\n")
    
    print("All scenes written.")
