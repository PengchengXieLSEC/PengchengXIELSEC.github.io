from math import *
from numpy import *


def Sch_orth(d, u):  # d,u为array类型
    n = len(d)
    if n > 1:
        DD = dot(d, d)
        UU = dot(u, u)
        DU = dot(d, u)
        s = (DD * u - DU * d) / sqrt(abs(DD * UU - DU * DU))
    else:
        s = -d
    return s


def Q(x):  # c为浮点型,g、x0、gamma、Gammy为array类型,m为整型,
    global c, g, Gamma, gamma, x0, Xn, m
    G = Gamma
    n = len(g)
    for i in range(0, m):
        G = G + gamma[i] * (dot(Xn[:, i] - x0, Xn[:, i] - x0))
    value = c + dot((x - x0), g) + 0.5 * dot(dot((x - x0), G), (x - x0))
    return value


def NEWUOAMethod(Fobj, M, N, xbeg, rhobeg, rhoend):
    print(Fobj,M,N,xbeg,rhobeg,rhoend)
    global Xn, Fn, n, m, F_times, rho_beg, rho_end, x0, opt, xb, c, g, Gamma, gamma, H
    global F, rho, delta, Krho, D3, QF3, CRVMIN, d, NORMD, Qnew
    global RATIO, MOVE, w, Hw, beta, Fnew, DIST, XXopt, NXX
    global  order
    order = array([])
    F = Fobj
    m = M
    n = N
    xb = xbeg
    rho_beg = rhobeg
    rho_end = rhoend
    # ============= global end ==========================
    NEWUOAStep1()
    Fopt = Fn[opt]
    xopt = Xn[:, opt]
    return Fopt, xopt


def NEWUOAStep1():
    global Xn, Fn, n, m, F_times, rho_beg, x0, opt, xb, c
    global g, Gamma, gamma, H
    global F, rho, delta
    global Krho, D3, QF3
    global order
    order = append(order,1)
    n = 2  # len(g)
    m = 5
    xb = array([10.0, 10.0])
    rho_beg = 1.0

    gamma = array([0.0] * m)  # array类型
    Xn = zeros((n, m))  # array类型
    Fn = array([0.0] * m)  # array类型
    x0 = xb  # array类型
    Xn[:, 0] = x0  # array类型
    Fn[0] = F(Xn[:, 0])
    I = eye(n)  # array类型
    Gamma = zeros((n, n))  # array类型
    c = Fn[0]  # 浮点型
    g = array([0.0] * n)  # array类型
    rho = rho_beg  # 浮点型
    delta = rho  # 浮点型
    Krho = 0  # 整型
    D3 = array([0.0] * 3)  # array类型
    QF3 = array([0.0] * 3)  # array类型
    # ============= global end ==========================
    # #生成插值点
    Delta = array([0.0] * n)  # array类型
    if m >= 2 * n + 1:
        for i in range(0, n):
            Xn[:, i + 1] = x0 + rho * I[:, i]
            Fn[i + 1] = F(Xn[:, i + 1])
            Xn[:, i + n + 1] = x0 - rho * I[:, i]
            Fn[i + n + 1] = F(Xn[:, i + n + 1])
            if Fn[i + n + 1] >= Fn[i + 1]:
                Delta[i] = 1
            else:
                Delta[i] = -1
            g[i] = (Fn[i + 1] - Fn[i + n + 1]) / (2 * rho)
            Gamma[i, i] = (Fn[i + 1] + Fn[i + n + 1] - 2 * Fn[0]) / (rho * rho)

        for i in range(2 * n + 1, m):
            j = math.floor(((i + 1) - n - 2) / n)
            p = (i + 1) - n - 1 - j * n
            if p + j > n:
                q = p + j - n
            else:
                q = p + j
            Xn[:, i] = x0 + rho * Delta[p] * I[:, p - 1] + rho * Delta[q - 1] * I[:, q - 1]
            Fn[i] = F(Xn[:, i])
            Gamma[p - 1, q - 1] = (Fn[0] - Fn[1 + (p - 1) + (1 - Delta[p - 1]) * n / 2] - Fn[1 +
                                                                                             (q - 1) + (1 - Delta[
                q - 1]) * n / 2] + Fn[i]) / (
                                          Delta[p - 1] * Delta[q - 1] * rho * rho)
            Gamma[q - 1, p - 1] = Gamma[p - 1, q - 1]
    else:
        for i in range(0, n):
            Xn[:, i + 1] = x0 + rho * I[:, i]
            Fn[i + 1] = F(Xn[:, i + 1])
        for i in range(n, m - 1):
            Xn[:, i + 1] = x0 - rho * I[:, i - n]
            Fn[i + 1] = F(Xn[:, i + 1])
            g[i - n] = (Fn[i - n + 1] - Fn[i + 1]) / (2 * rho)
            Gamma[i - n, i - n] = (Fn[i - n + 1] + Fn[i + 1] - 2 * Fn[0]) / (rho * rho)
        for i in range(m - 1, 2 * n):
            g[i - n] = (Fn[i - n + 1] - Fn[0]) / (rho)
    F_times = m

    # #计算W的逆H
    Theta = zeros((n + 1, m))
    Theta[0, 0] = 1
    Upsilon = zeros((n + 1, n + 1))
    if m >= 2 * n + 1:
        for i in range(1, n + 1):
            Theta[i, i] = 1 / (2 * rho)
            Theta[i, i + n] = -1 / (2 * rho)
    else:
        for i in range(1, m - n):
            Theta[i, i] = 1 / (2 * rho)
            Theta[i, i + n] = -1 / (2 * rho)
        for i in range(m - n, n + 1):
            Theta[i, 0] = -1 / rho
            Theta[i, i] = 1 / rho
            Upsilon[i, i] = -0.5 * rho * rho
    Z = zeros((m, m - n - 1))
    s = ones(m - n - 1)
    if m <= 2 * n + 1:
        for k in range(0, m - n - 1):
            Z[0, k] = -math.sqrt(2) / (rho * rho)
            Z[k + 1, k] = -Z[0, k] / 2
            Z[k + n + 1, k] = Z[k + 1, k]
    else:
        for k in range(0, n):
            Z[0, k] = -math.sqrt(2) / (rho * rho)
            Z[k + 1, k] = -Z[0, k] / 2
            Z[k + n + 1, k] = Z[k + 1, k]
        for k in range(n, m - n - 1):
            i = k + n + 1
            j = math.floor(((i + 1) - n - 2) / n)
            p = (i + 1) - n - 1 - j * n
            if p + j > n:
                q = p + j - n
            else:
                q = p + j
            if Delta[p - 1] == 1:
                pt = p + 1
            else:
                pt = p + 1 + n
            if Delta[q - 1] == 1:
                qt = q + 1
            else:
                qt = q + 1 + n
            Z[0, k] = 1 / (rho * rho)
            Z[pt - 1, k] = -Z[0, k]
            Z[qt - 1, k] = Z[pt - 1, k]
            Z[k + n + 1, k] = Z[0, k]
    H = zeros((m + n + 1, m + n + 1))  # array类型
    H[0: m, 0: m] = dot(Z, Z.T)
    H[0: m, m: m + n + 1] = Theta.T
    H[m: m + n + 1, 0: m] = Theta
    H[m: m + n + 1, m: m + n + 1] = Upsilon
    opt = where(Fn == min(Fn))
    opt = opt[0]
    opt = opt[0]
    global Test
    Test = zeros((10000, n + 1))
    Test[F_times - m + 1, :] = hstack((Xn[:, opt], Fn[opt]))
    NEWUOAStep2()


def NEWUOAStep2():  # gamma、XX、u为array类型
    global n, m, g, Gamma, gamma, x0, Xn, Fn, opt, delta, CRVMIN, d, NORMD, Qnew

    global order
    order = append(order,2)

    XX = zeros((n, m))

    for i in range(m):
        XX[:, i] = Xn[:, i] - x0

    def DDQ(u):  # gamma、XX、u为array类型
        eta = gamma * dot(XX.T, u)
        R = dot(Gamma, u)
        for ii in range(0, m):
            R = R + eta[ii] * XX[:, ii]
        return R

    CRVMIN = 0  # 默认设置为0
    xopt = Xn[:, opt]
    Fopt = Fn[opt]
    d = array([0.0] * n)
    NORMD = 0
    S = zeros((n, n))  # 每一步的搜索方法
    sDDQs = array([0.0] * n)  # 保存以备使用
    Q0 = Fopt
    Qold = Fopt
    u = xopt - x0
    DQ = g + DDQ(u)
    DQ0 = DQ
    k = 1  # 计算的步数
    S[:, k - 1] = -DQ
    Ns = linalg.norm(S[:, k - 1])
    NDQ0 = Ns

    if (Ns <= 10 ** -8):  # xopt已经是模型的驻点了
        d = zeros([0.0] * n)
        NORMD = 0
        Qnew = Q0
    else:  # 此时才能进行正常的迭代
        a = Ns * Ns
        b = 2 * dot(d, S[:, k - 1])
        c = NORMD ** 2 - delta ** 2
        alpha = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)
        DDQs = DDQ(S[:, k - 1])

        sDDQs[k] = dot(S[:, k - 1], DDQs)
        N2DQxd = (NDQ0) ** 2
        if (N2DQxd < (alpha * sDDQs[k - 1])):
            alpha = N2DQxd / sDDQs[k - 1]

        # 已经确定好alpha与s
        d = d + alpha * S[:, k - 1]  # 新的d
        NORMD = linalg.norm(d)
        Qnew = Qold + alpha * dot(S[:, k], DQ) + sDDQs[k - 1] * alpha * alpha / 2
        k = k + 1  # 下一步开始
        NDQold = NDQ0
        DQ = DQ + alpha * DDQs  # 更新到当前点的梯度。当前点的梯度能由之前的信息获得。
        NDQnew = linalg.norm(DQ)
        # 开始迭代
        while ((k <= n) and (NDQnew > 0.01 * NDQ0) and ((Qold - Qnew) > 0.01 * ((Q0 - Qnew)))):  # 进入循环的三个条件
            Qold = Qnew
            beta = (NDQnew / NDQold) ** 2
            S[:, k - 1] = -DQ + beta * S[:, k - 2]
            Ns = linalg.norm(S[:, k - 1])
            if (Ns <= 10 ** -8):  # 此时的s实在是太小了，无法继续搜索，故跳出
                Qnew = Qold
                k = k + 1
                break
            else:
                a = Ns * Ns
                b = 2 * dot(d, S[:, k - 1])
                c = NORMD ** 2 - delta ** 2
                alpha = (-b + sqrt(b * b - 4 * a * c)) / (2 * a)
                DDQs = DDQ(S[:, k])
                sDDQs[k - 1] = dot(S[:, k - 1], DDQs)
                N2DQxd = NDQnew ** 2  # #||DQ(xopt+d_(j-1))||**2
                if N2DQxd < alpha * sDDQs[k - 1]:
                    alpha = N2DQxd / sDDQs[k - 1]

                # 已经确定好alpha与s
                d = d + alpha * S[:, k - 1]  # 新的d
                NORMD = linalg.norm(d)
                Qnew = Qold + alpha * dot(S[:, k - 1], DQ) + sDDQs[k - 1] * alpha * alpha / 2
                k = k + 1  # 下一步开始
                NDQold = NDQnew
                DQ = DQ + alpha * DDQs  # 更新到当前点的梯度。当前点的梯度能由之前的信息获得。
                NDQnew = linalg.norm(DQ)

    if abs(NORMD - delta) >= 0.01 * delta:
        if k == 1:
            CRVMIN = 0
        else:
            CRV = zeros((k - 1, 1))
            for i in range(0, k - 1):
                CRV[i] = sDDQs[i] / dot(S[:, i], S[:, i])
            CRVMIN = min(CRV)

    else:
        CRVMIN = 0
        DQd = dot(d, DQ)
        if (NDQnew > 0.01 * NDQ0) and (DQd > -0.99 * NDQnew * NORMD) and (k <= n):  # 满足这个三个条件是，需要额外的迭代
            theta = array([0] * 50)
            for i in range(0, 50):
                theta[i] = 2 * ((i + 1) - 1) * pi / 50

            Qold = Qnew
            s = Sch_orth(d, DQ)
            DDQs = DDQ(s)
            DQDQ0 = DQ - DQ0
            sDDQs[k - 1] = dot(s, DDQs)
            Qtheta = array([0.0] * 50)
            for i in range(0, 50):
                costheta = cos(theta[i])
                sintheta = sin(theta[i])
                Qtheta[i] = Q0 + dot(costheta * d + sintheta * s, DQ0) + dot(
                    0.5 * costheta * costheta * d + costheta * sintheta * s, DQDQ0) + 0.5 * sintheta * sintheta * sDDQs[
                                k - 1]

            Qnew = min(Qtheta)
            kd = where(Qtheta == Qnew)
            kd = kd[0]
            kd = kd[0]
            d = cos(theta[kd]) * d + sin(theta[kd]) * s  # 新的d
            NORMD = linalg.norm(d)
            k = k + 1  # 下一步开始
            NDQold = NDQnew
            DQ = (1 - cos(theta[kd])) * DQ0 + cos(theta[kd]) * DQ + sin(theta[kd]) * DDQs  # 更新到当前点的梯度。当前点的梯度能由之前的信息获得。
            NDQnew = linalg.norm(DQ)
            while ((k <= n) and ((Qold - Qnew) > 0.01 * ((Q0 - Qnew)))):
                Qold = Qnew
                s = Sch_orth(d, DQ)
                DDQs = DDQ(s)
                DQDQ0 = DQ - DQ0
                sDDQs[k - 1] = dot(s, DDQs)
                Qtheta = array([0.0] * 60)
                for i in range(0, 50):
                    costheta = cos(theta[i])
                    sintheta = sin(theta[i])
                    Qtheta[i] = Q0 + dot(costheta * d + sintheta * s, DQ0) + dot(
                        0.5 * costheta * costheta * d + costheta * sintheta * s, DQDQ0) + 0.5 * sintheta * sintheta * \
                                sDDQs[k - 1]

                Qnew = min(Qtheta)
                kd = where(Qtheta == Qnew)
                kd = kd[0]
                kd = kd[0]
                d = cos(theta[kd]) * d + sin(theta[kd]) * s  # 新的d
                NORMD = linalg.norm(d)
                k = k + 1  # 下一步开始
                NDQold = NDQnew
                DQ = (1 - cos(theta[kd])) * DQ0 + cos(theta[kd]) * DQ + sin(
                    theta[kd]) * DDQs  # 更新到当前点的梯度。当前点的梯度能由之前的信息获得。
                NDQnew = linalg.norm(DQ)

    NEWUOAStep3()


def NEWUOAStep3():
    global NORMD, rho

    global order
    order = append(order,3)

    if NORMD > 0.5 * rho:
        NEWUOAStep4()
    else:
        NEWUOAStep14()


def NEWUOAStep4():
    global m, n, RATIO, Fn, Xn, F, F_times, delta, opt, d
    global NORMD, rho, MOVE, Qnew, H, x0, w, Hw, beta, Fnew, Krho
    global DandR
    global order
    order = append(order,4)
    Fopt = Fn[opt]
    xopt = Xn[:, opt]
    xnew = xopt + d
    Fnew = F(xnew)
    F_times = F_times + 1
    Krho = Krho + 1
    dQ = Fopt - Qnew
    dF = Fopt - Fnew
    DDQ = Q(Xn[:, opt]) - Q(xnew)
    if dQ <= 0:
        if dF > 0:
            RATIO = 0
        else:
            RATIO = -1
    else:
        RATIO = dF / dQ
    if RATIO <= 0.1:
        deltaint = 0.5 * NORMD
    else:
        if RATIO <= 0.7:
            deltaint = max(NORMD, 0.5 * delta)
        else:
            deltaint = max(2 * NORMD, 0.5 * delta)
    if deltaint > 1.5 * rho:
        delta = deltaint
    else:
        delta = rho
    T = array(range(0, m))
    if dF > 0:
        Case = 1
        xstar = xopt + d
    else:
        xstar = xopt
        T = delete(T, opt)
        Case = -1
    w = zeros(m + n + 1)
    dxx0 = xnew - x0
    for i in range(0, m):
        w[i] = 0.5 * dot(Xn[:, i] - x0, power(dxx0, 2))
    w[m] = 1
    w[m + 1:m + n + 1] = dxx0
    Hw = dot(H, w)
    beta = 0.5 * ((dot(dxx0, dxx0)) ** 2) - dot(w, Hw)
    Sigma = zeros(m)
    Weight = zeros(m)
    M1 = max(0.1 * delta, rho)
    if Case > 0:
        for i in range(0, m):
            alpha = H[i, i]
            tau = Hw[i]
            Sigma[i] = alpha * beta + tau * tau
            Weight[i] = max(1, (linalg.norm(Xn[:, i] - xstar) / M1) ** 6)
        tstar = where(Weight * abs(Sigma) == max(Weight * abs(Sigma)))
        tstar = tstar[0]
        tstar = tstar[0]
        MOVE = tstar
    else:
        for i in range(0, m-1):
            print(T)
            print(H)
            alpha = H[T[i], T[i]]
            tau = Hw[T[i]]
            Sigma[T[i]] = alpha * beta + tau * tau
            Weight[T[i]] = max(1, (linalg.norm(Xn[:, T[i]] - xstar) / M1) ** 6)
        M2 = max(Weight * abs(Sigma))
        tstar = where(Weight * abs(Sigma) == M2)
        tstar = tstar[0]
        tstar = tstar[0]
        if M2 <= 1:
            MOVE = 0
        else:
            MOVE = tstar
    # DandR = [DandR, delta, RATIO, E]

    NEWUOAStep5()


def NEWUOAStep5():
    global m, n, d, MOVE, Krho, D3, QF3, NORMD, Xn, Fn, opt, x0
    global Qnew, Fnew, c, g, Gamma, gamma, H, w, Hw, beta, RATIO
    global Test, F_times
    global order
    order = append(order,5)
    K = Krho % 3 + 1
    D3[K - 1] = NORMD
    QF3[K - 1] = abs(Qnew - Fnew)
    t = MOVE
    if t > 0:
        XX = zeros((n, m))
        for i in range(m):
            XX[:, i] = Xn[:, i] - x0
        s = XX[:, opt]
        NS2 = dot(s, s)
        xnew = Xn[:, opt] + d
        if (NORMD * NORMD <= (0.001 * NS2)):
            xav = 0.5 * (x0 + Xn[:, opt])
            x0 = Xn[:, opt]
            v = zeros(n)
            for i in range(0, m):
                v = v + gamma[i] * (Xn[:, i] - xav)
            vs = v * s
            Gamma = Gamma + vs + vs
            for i in range(m):
                XX[:, i] = Xn[:, i] - x0
            eta = gamma * dot(XX, s)
            DDQs = dot(Gamma, s)
            for i in range(0, m):
                DDQs = DDQs + eta[i] * XX[:, i]
            g = g + DDQs
            c = Fn[opt]
            X = zeros((n + 1, m))
            X[0, :] = ones(m)
            X[1: n + 1, :] = XX[:, :]
            A = zeros((m, m))
            for i in range(0, m):
                for j in range(0, m):
                    A[i, j] = 0.5 * dot(XX[:, i], XX[:, j]) ** 2
            W = zeros((m + n + 1, m + n + 1))
            W[0: m, 0: m] = A[:, :]
            W[0: m, m - 1: m + n + 1] = X.T[:, :]
            W[m: m + n + 1, 0: m] = X[:, :]
            H = linalg.inv(W)

            w = zeros(m + n + 1)
            dxx0 = xnew - x0
            for i in range(0, m + 1):
                w[i] = 0.5 * (dot(Xn[:, i] - x0, dxx0) ** 2)
            w[m] = 1
            w[m + 1: m + n + 1] = dxx0[:]
            Hw = dot(H, w)
            beta = 0.5 * (dot(dxx0, dxx0) ** 2) - dot(w, Hw)
        alpha = H[t - 1, t - 1]
        tau = Hw[t - 1]
        sigma = alpha * beta + tau ** 2
        et = zeros(n + m + 1)
        et[t - 1] = 1
        eHw = et - Hw
        H = H + (alpha * (eHw.reshape(n + m + 1, 1) * eHw) - beta * H[:, t - 1].reshape(n + m + 1, 1) * H[t - 1, :] + tau * (H[:, t - 1].reshape(n + m + 1, 1) * eHw + eHw.reshape(n + m + 1, 1) * H[t, :])) / sigma
        if (RATIO >= 0.5):
            C = Fnew - Qnew
            Lcg = C * H[:, t - 1]
        else:
            R = zeros(m + n + 1)
            for i in range(0, m):
                if not i == t:
                    R[i] = Fn[i] - Q(Xn[:, i])
                else:
                    R[i] = Fnew - Q(xnew)
            Lcg = dot(H, R)
        lambda_ = Lcg[0:m]
        dc = Lcg[m]
        dg = Lcg[m + 1: m + n + 1]
        c = c + dc
        g = g + dg
        Gamma = Gamma + gamma[t] * dot(XX[:, t], XX[:, t])
        for i in range(0, m):
            if not i == t - 1:
                print(gamma[i])
                print(lambda_[i])
                gamma[i] = gamma[i] + lambda_[i]
            else:
                gamma[i] = lambda_[i]

        if Fnew < Fn[opt]:
            opt = t - 1
            Fn[t - 1] = Fnew
            Xn[:, t - 1] = xnew
        else:
            Fn[t - 1] = Fnew
            Xn[:, t - 1] = xnew

    Test[F_times - m, :] = append(Xn[:, opt], Fn[opt])

    NEWUOAStep6()


def NEWUOAStep6():
    global order
    order = append(order,6)
    global RATIO
    if RATIO >= 0.1:
        NEWUOAStep2()
    else:
        NEWUOAStep7()


def NEWUOAStep7():
    global order
    order = append(order,7)
    global DIST, MOVE, opt, Xn, n, m, XXopt, NXX
    DIST = 0
    XXopt = zeros((n, m))
    NXX = zeros(m)
    for i in range(0, m):
        XXopt[:, i] = Xn[:, i] - Xn[:, opt]
        NXX[i] = sqrt(dot(XXopt[:, i], XXopt[:, i]))
        if NXX[i] > DIST:
            DIST = NXX[i]
            MOVE = i
    NEWUOAStep8()


def NEWUOAStep8():
    global order
    order = append(order,8)
    global DIST, delta
    if DIST >= 2 * delta:
        NEWUOAStep9()
    else:
        NEWUOAStep10()


def NEWUOAStep9():
    global order
    order = append(order,9)
    global F, d, NORMD, beta, w, Hw, rho, delta, DIST, H
    global Xn, MOVE, opt, x0, n, m, RATIO, Fnew, Qnew, F_times, Krho, XXopt, NXX, XX0
    deltah = max(min([0.1 * DIST, 0.5 * delta]), rho)
    DELSQ = deltah * deltah
    N = n
    t = MOVE
    xt = Xn[:, t]
    xopt = Xn[:, opt]
    Lambda = diag(H[0:m, t - 1])
    c = H[m + 1, t]
    g = H[m + 2:m + n + 1, t]
    XX0 = zeros((n,m))
    for i in range(0,m):
        XX0[:,i] = Xn[:,i]-x0
    print("XXXXXXXXXXXXXXXXXX0000000")
    print(XX0)
    print(Lambda)
    G = dot(dot(XX0 , Lambda) , XX0.T)
    Dl = g + dot(G , XX0[:, opt])

    def eta(u):  # u为array类型
        return H[0:m, t - 1] * dot(XX0.T, u)#5*1   5*1

    def DDlu(u):  # u为array类型
        return dot(XX0, eta(u).T)#2*5

    def DENFunction(d, v, t, C):
        global Xn, m, n, XX0, opt, x0, H
        Xnew = Xn[:, opt] + d
        xx0 = Xnew - x0
        w = zeros(m + n + 1)
        for i in range(0, m):
            w[i] = 0.5 * (dot(XX0[:, i], xx0) ** 2)
        w[m] = 1
        w[m + 1: m + n + 1] = xx0
        alpha = H[t, t]
        wv = w - v
        Hwv = dot(H , wv)
        DEN = alpha * (0.5 * (dot(xx0, xx0) ** 2) - dot(XX0[:, opt], xx0) ** 2 + C) - alpha * dot(wv, Hwv) + Hwv[t]
        return DEN

    def DENFALL(d, v, t, C):
        global Xn, m, n, XX0, opt, x0, H
        Xnew = Xn[:, opt] + d
        xx0 = Xnew - x0
        w = zeros(m + n + 1)
        for i in range(0, m):
            w[i] = 0.5 * (dot(XX0[:, i], xx0) ** 2)
        w[m] = 1
        w[m + 1: m + n + 1] = xx0
        alpha = H[t, t]
        wv = w - v
        Hwv = dot(H , wv)
        NX0 = dot(xx0, xx0)
        DEN = alpha * (0.5 * (NX0 ** 2) - dot(XX0[:, opt], xx0) ** 2 + C) - alpha * dot(wv, Hwv) + Hwv[t]

        Hw = dot(H , w)
        beta = 0.5 * (NX0 ** 2) - dot(w, Hw)
        tau = Hw[t]
        eta1 = Hwv[0:m]
        eta2 = Hwv[m + 1:m + n + 1]
        DDEN2 = zeros(n)
        DDEN3 = zeros(n)
        for i in range(0, m):
            print("22222",eta1[i])
            DDEN2 = DDEN2 + ((tau * H[t, i] - alpha * eta1[i] * dot(xx0, XX0[:, i])) * XX0[:, i])
        DDEN2 = 2 * DDEN2
        for i in range(0, n):
            DDEN3[i] = 2 * (tau * H[t, i + m + 1] - alpha * eta2[i])
        DDEN = 2 * alpha * (NX0 * d + dot(dot(d, xx0), XX0[:, opt])) + (DDEN2) + (DDEN3)
        return DEN, w, Hw, beta, DDEN

    xx = XXopt[:, t]
    d = deltah * (xx / linalg.norm(xx))
    GD = DDlu(d)
    DD = dot(d,d)
    GG = dot(Dl,Dl)
    SP = dot(d,Dl)
    DHD = dot(d,GD)
    if SP * DHD <= 0:
        d = -d
        GD = -GD
        lxd = abs(-SP + 0.5 * DHD)
    else:
        lxd = abs(SP + 0.5 * DHD)
    TEMP = 0
    if (SP * SP >= 0.99 * DD * GG) or (GG * DELSQ <= 0.01 * lxd * lxd):
        TEMP=1
    S=Dl+TEMP * GD
    K1=0
    CTH=zeros((49, 1))
    STH=zeros((49, 1))
    ANGLE=zeros((49, 1))
    dPi=2 * pi / 50
    for i in range(0, 49):
        ANGLE[i] = dPi * i
        CTH[i] = cos(ANGLE[i])
        STH[i] = sin(ANGLE[i])
    while K1 <= N:
        K1 = K1 + 1
        DD = dot(d,d)
        SP = dot(d,S)
        SS = dot(S,S)
        TEMP = DD * SS - SP * SP
        if (TEMP <= 10**-8 * DD * SS):
            break
        DENOM = sqrt(TEMP)
        S = (DD * S - SP * d) / DENOM
        W = DDlu(S)
        CF1 = 0.5 * dot(S,W)
        CF2 = dot(d,Dl)
        CF3 = dot(S,Dl)
        CF4 = 0.5 * dot(d,GD)-CF1
        CF5 = dot(S,GD)
        TAUBEG = CF1 + CF2 + CF4
        TAUMAX = TAUBEG
        TAUOLD = TAUBEG
        ISAVE = 0
        TAU = 0
        TEMPA = 0
        TEMPB = 0
        for i in range(0,49):
            TAU = CF1 + (CF2 + CF4 * CTH[i]) * CTH[i] + (CF3 + CF5 * CTH[i]) * STH[i]
            if (abs(TAU) >= abs(TAUMAX)):
                TAUMAX = TAU
                ISAVE = i
                TEMPA = TAUOLD
            else:
                if (i == (ISAVE + 1)):
                    TEMPB = TAU
            TAUOLD = TAU
        if (ISAVE == 0):
            TEMPA = TAU
        if (ISAVE == 49):
            TEMPB = TAUBEG
        STEP = 0
        if (TEMPA != TEMPB):
            TEMPA = TEMPA - TAUMAX
            TEMPB = TEMPB - TAUMAX
            STEP = 0.5 * (TEMPA - TEMPB) / (TEMPA + TEMPB)
        angle = dPi * (ISAVE + STEP)
        TAU = CF1 + (CF2 + CF4 * cos(angle)) * cos(angle) + (CF3 + CF5 * cos(angle)) * sin(angle)
        d = cos(angle) * d + sin(angle) * S
        GD = cos(angle) * GD + sin(angle) * W
        S = Dl + GD
        if (abs(TAU) <= 1.1 * abs(TAUBEG)):
            break
    NORMD = deltah
    xnew = xopt + d
    alpha = H[t, t]
    w = zeros(m + n + 1)
    xx0 = xnew - x0
    for i in range(0,m):
        w[i] = 0.5 * (dot(XX0[:, i],xx0)**2)
    w[m] = 1
    w[m + 1: m + n + 1]=xx0
    Hw = dot(H , w)
    beta = 0.5 * (linalg.norm(xx0) ** 4) - dot(w,Hw)
    tau = Hw[t]
    tau2 = tau * tau

    if abs(alpha * beta + tau2) <= 0.8 * tau2:
        S = XXopt[:, t]
        DD = dot(d,d)
        DS = dot(d,S)
        SS = dot(S,S)
        v = zeros(m + n + 1)
        for i in range(0,m):
            v[i] = 0.5 * (dot(XX0[:, i],XX0[:,opt])**2)
        v[m] = 1
        v[m + 1: m + n + 1]=XX0[:, opt]
        NCN = 0.5 * (dot(XX0[:, opt],XX0[:,opt])**2)

        def DEN(d):
            return DENFunction(d, v, t, NCN)

        Weight = DS * DS / (DD * SS)
        Kt = t
        if (Weight >= 0.99):
            for i in range(0,m):
                if i != opt:
                    DSTEMP = dot(XXopt[:, i],d)
                    SSTEMP = NXX[i] ** 2
                    Weightnew = DSTEMP * DSTEMP / (DD * SSTEMP)
                    if Weightnew <= Weight:
                        Kt = i
                        Weight = Weightnew
                        DS = DSTEMP
                        SS = SSTEMP
            S = XXopt[:, Kt]
        SSDEN = DD * SS - DS * DS
        K2 = 0
        TEMP = 1 / sqrt(SSDEN)
        S = TEMP * (DD * S - DS * d)
        while (K2 <= N):
            DENTEST = zeros(50)
            print(DEN(d))
            DENTEST[0] = abs(DEN(d))
            for i in range(0,49):
                dtest = CTH[i] * d + STH[i] * S
                DENTEST[i + 1] = abs(DEN(dtest))
            DENMAX = max(DENTEST)
            ISAVE = where(DENTEST == DENMAX)
            ISAVE = ISAVE[0]
            ISAVE = ISAVE[0]
            if ISAVE == 0:
                TEMPA = DENTEST[49]
                TEMPB = DENTEST[1]
            else:
                if ISAVE == 49:
                    TEMPA = DENTEST[48]
                    TEMPB = DENTEST[0]
                else:
                    TEMPA = DENTEST[ISAVE - 1]
                    TEMPB = DENTEST[ISAVE + 1]
            STEP = 0
            if (TEMPA != TEMPB):
                TEMPA = TEMPA - DENMAX
                TEMPB = TEMPB - DENMAX
                STEP = 0.5 * (TEMPA - TEMPB) / (TEMPA + TEMPB)
            angle = dPi * (ISAVE + STEP)
            d = cos(angle) * d + sin(angle) * S
            DENNEW, w, Hw, beta, S = DENFALL(d, v, t, NCN)
            if K2 == 0:
                DENOLD = abs(DENNEW)
            else:
                if (abs(DENNEW) <= 1.1 * DENOLD):
                    DENOLD = abs(DENNEW)
                    break
                else:
                    DENOLD = abs(DENNEW)
            DD = dot(d,d)
            DS = dot(d,S)
            SS = dot(S,S)
            SSDEN = DD * SS - DS * DS
            if (SSDEN >= 10**-8 * DD * SS):
                K2 = K2 + 1
                TEMP = 1 / sqrt(SSDEN)
                S = TEMP * (DD * S - DS * d)
            else:
                break
    xopt = Xn[:, opt]
    xnew = xopt + d
    Fnew = F(xnew)
    Qnew = Q(xnew)
    F_times = F_times + 1
    Krho = Krho + 1
    RATIO = 1
    NEWUOAStep5()


def NEWUOAStep10():
    global order
    order = append(order,10)
    global delta, RATIO, NORMD, rho
    M1 = max(NORMD, delta)
    if ((M1 - rho) / rho < pow(10, -6)) and (RATIO <= 0):
        NEWUOAStep11()
    else:
        NEWUOAStep2()


def NEWUOAStep11():
    global order
    order = append(order,11)
    global rho, rho_end
    if log10(rho) <= log10(rho_end):
        NEWUOAStep13()
    else:
        NEWUOAStep12()


def NEWUOAStep12():
    global order
    order = append(order,12)
    global rho, delta, Krho

    delta = 0.5 * rho
    rho = rho / 10
    Krho = 0
    NEWUOAStep2()


def NEWUOAStep13():
    global order
    order = append(order,13)
    global d, D3, rho, Xn, opt, Fnew, F, F_times
    if D3[0] < 0.5 * rho:
        xopt = Xn[:, opt]
        Fnew = F(xopt + d)
        F_times = F_times + 1
        print("00000000000000000000000")
        print(F_times)


def NEWUOAStep14():
    global order
    order = append(order,14)
    global CRVMIN, D3, QF3, rho, Krho
    if Krho >= 3:

        TEMP = 0.125 * CRVMIN * rho * rho
        if max(QF3) <= TEMP and max(D3) <= rho:
            NEWUOAStep15()
            NEWUOAStep11()

    else:
        NEWUOAStep15()


def NEWUOAStep15():
    global order
    order = append(order,15)
    # #NEWUOAStep15 对应NEWUOA算法的第十五步

    global RATIO, delta, rho
    delta = 0.1 * delta
    RATIO = -1
    if delta <= 1.5 * rho:
        delta = rho
    NEWUOAStep7()


def F(x):  # x为array类型
    return linalg.norm(x) ** 2


xbeg = array([100.0,100.0])
N = len(xbeg)
M = 2 * N + 1
rhobeg = 1.0
rhoend = 10 ** -6
print("11111")
NEWUOAMethod(F, M, N, xbeg, rhobeg, rhoend)
