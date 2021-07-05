STEP1:在桌面上单击一次文件xpc，点击此时桌面左上方出现的“前往”按钮，->点击“个人”->在搜索框中输入“gjbdbs”->单击一次搜索得到的文件gjbdbs，此时正下方会出现该文件所在的路径，->双击文件gjbdbs的上层文件python3.8->打开文件site-packages->打开文件夹susd_lqr->打开文件夹dynamics->dtnamic文件夹里面的linear_12D.py是我们要用到的辅助文件->返回上一级文件夹susd_lqr->再返回上一级文件夹site-packages，site-packages文件夹里面的yinzi_pdfo.py是我们要执行的第一个py文件->打开文件夹susd_lqr->打开文件夹Compare_LQR->Compare_LQR文件夹里面的Compare_LQR.py是我们要执行的py文件。
STEP2:在桌面正下方点击绿圈图标ANACONDA NAVIGATOR，->点击该软件左侧菜单栏中的Home-选择Application on miniconda3 Channel->打开python代码运行环境平台spyder5.0.3->将待执行py文件yinzi_pdfo.py，Compare_LQR和linear_12D.py拖入spyder界面，先运行yinzi_pdfo.py（点击左上方绿色三角形图标，运行程序），将linear_12D.py里的system函数里矩阵A和矩阵B的取值分别改为：

        A = np.array([[-2.5, 1.2, 4.3, 0.1], 
                      [0.97, -10.3, 0.4, -6.1],
                      [-9.2, 1.1, -4.9, 0.3],
                      [1.1, 0.9, -3.4, -0.9]])
        B = np.array([[ 1.1,  0.4, -0.2, 0], 
                      [-3.2,  1.4,  0.0, 0],
                      [-0.8,  0.1,  3.0, 0],
                      [-1.1, -0.9,  5.2, 1]])
按下commend+s保存。
再在Comparer_LQR里面的第65行，第66行改为：
    nb_states = 4
    nb_actions = 4
将第45行改为
    N_problem = 100
按下commend+s保存
运行Compare_LQR
的所有图（100张问题图+1个perf图+1个data图）都会被存在文件Compare_LQR文件夹中。