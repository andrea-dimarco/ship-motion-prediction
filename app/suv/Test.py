from Traiettoria_Quad import Traiettoria_Scitech

test_traiettoria = Traiettoria_Scitech()
time_test = 2.0
result = test_traiettoria.TrajectoryEval(time_test)
print(f"Risultato della traiettoria al tempo {time_test} s: {result}")