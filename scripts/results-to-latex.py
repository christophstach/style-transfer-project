def file_to_json(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    results = [
        {
            param.strip().split('=')[0]: param.strip().split('=')[1]
            for param in str(line).replace('\n', '').split(',')
        }
        for line in lines
        if line and line != 'GPU\n' and line != 'CPU\n'
    ]

    cleaned_results = []
    for result in results:
        image_size_wh = result['image_size'].split('x')

        cleaned_results.append({
            'bottleneck_size': int(result['bottleneck_size']),
            'channel_multiplier': int(result['channel_multiplier']),
            'device_type': str(result['device_type']),
            'iterations': int(result['iterations']),
            'result': float(result['result']),
            'image_size': (int(image_size_wh[0]), int(image_size_wh[1]))
        })

    return cleaned_results


xps_640x480 = file_to_json('./results/xps-640x480.txt')
xps_1024x768 = file_to_json('./results/xps-1024x768.txt')
xps_1920x1080 = file_to_json('./results/xps-1920x1080.txt')
jet_640x480 = file_to_json('./results/jetson-640x480.txt')
jet_1024x768 = file_to_json('./results/jetson-1024x768.txt')
jet_1920x1080 = file_to_json('./results/jetson-1920x1080.txt')

out_str = ''
for i in range(16):
    err = '\\textcolor{danger}{nicht durchführbar}'
    spc = '                               '

    xps_cuda_result = f"{xps_1920x1080[i]['result']:.5f}{spc}" if xps_1920x1080[i]['result'] != -1 else err
    xps_cpu_result = f"{xps_1920x1080[16 + i]['result']:.5f}{spc}" if xps_1920x1080[16 + i]['result'] != -1 else err
    jet_cuda_result = f"{jet_1920x1080[i]['result']:.5f}{spc}" if jet_1920x1080[i]['result'] != -1 else err
    jet_cpu_result = f"{jet_1920x1080[16 + i]['result']:.5f}{spc}" if jet_1920x1080[16 + i]['result'] != -1 else err

    out_str += f'Network {i + 1:2}'
    out_str += f' & {xps_cpu_result.replace(".", ",")}'
    out_str += f' & {xps_cuda_result.replace(".", ",")}'
    out_str += f' & {jet_cpu_result.replace(".", ",")}'
    out_str += f' & {jet_cuda_result.replace(".", ",")}'
    out_str += ' \\\\ \\hline\n'

out_str += '\n\n\n'

for i in range(16):
    err = '\\textcolor{danger}{nicht durchführbar}'
    spc = '                               '

    xps_cuda_result = f"{xps_1024x768[i]['result']:.5f}{spc}" if xps_1024x768[i]['result'] != -1 else err
    xps_cpu_result = f"{xps_1024x768[16 + i]['result']:.5f}{spc}" if xps_1024x768[16 + i]['result'] != -1 else err
    jet_cuda_result = f"{jet_1024x768[i]['result']:.5f}{spc}" if jet_1024x768[i]['result'] != -1 else err
    jet_cpu_result = f"{jet_1024x768[16 + i]['result']:.5f}{spc}" if jet_1024x768[16 + i]['result'] != -1 else err

    out_str += f'Network {i + 1:2}'
    out_str += f' & {xps_cpu_result.replace(".", ",")}'
    out_str += f' & {xps_cuda_result.replace(".", ",")}'
    out_str += f' & {jet_cpu_result.replace(".", ",")}'
    out_str += f' & {jet_cuda_result.replace(".", ",")}'
    out_str += ' \\\\ \\hline\n'

out_str += '\n\n\n'

for i in range(16):
    err = '\\textcolor{danger}{nicht durchführbar}'
    spc = '                               '

    xps_cuda_result = f"{xps_640x480[i]['result']:.5f}{spc}" if xps_640x480[i]['result'] != -1 else err
    xps_cpu_result = f"{xps_640x480[16 + i]['result']:.5f}{spc}" if xps_640x480[16 + i]['result'] != -1 else err
    jet_cuda_result = f"{jet_640x480[i]['result']:.5f}{spc}" if jet_640x480[i]['result'] != -1 else err
    jet_cpu_result = f"{jet_640x480[16 + i]['result']:.5f}{spc}" if jet_640x480[16 + i]['result'] != -1 else err

    out_str += f'Network {i + 1:2}'
    out_str += f' & {xps_cpu_result.replace(".", ",")}'
    out_str += f' & {xps_cuda_result.replace(".", ",")}'
    out_str += f' & {jet_cpu_result.replace(".", ",")}'
    out_str += f' & {jet_cuda_result.replace(".", ",")}'
    out_str += ' \\\\ \\hline\n'

print(out_str)
