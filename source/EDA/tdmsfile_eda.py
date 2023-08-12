from nptdms import TdmsFile
from ..toolkit.plotClass import PlotManager

def main():

    prefix = "./data/"
    #read a tdms file
    filepath = prefix + "221108_nextgen/S206/test_05.tdms"
    tdms_file = TdmsFile(filepath)

    # 그룹 및 채널 목록 출력
    print("Groups in TDMS:", tdms_file.groups())
    print("Channels in Group:", tdms_file["LPData"].channels())

    # 데이터 읽기
    LPData = tdms_file["LPData"]["Channel"]
    LPData

    LPData_values = LPData[:]
    print("Data Values:", LPData_values)

    plotmanager = PlotManager(row=1, col=1, type='plot_numpy')

    plot_rawaudio_list = [
        {"content": LPData_values, "position": [0,0], "title":"TDMS"},
    ]

    plotmanager.drawPlot(plot_rawaudio_list, save_as_file="source/result/LPData_values.png")
    return 0

if __name__ == '__main__':
    main()

#파일 실행 명령: python -m source.EDA.tdmsfile_eda