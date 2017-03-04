execfile("cpi-master.py")
for n in [1, 5, 10, 100]:
    send(n)
    pi = recv()
    print(n, pi, abs(pi-math.pi))
send(0)
child.Disconnect()
