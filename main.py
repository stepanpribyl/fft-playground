from function import Function
from comparison import Comparison
from processor import Processor

fce = Function("Sines 1", 20, 50)
fce.add_sine(1, 6, 30, 1)
fce.add_sine(1, 9.3, 60, 1)
fce.add_noise(0.02)
fce.fft()
# fce.plot()

fce2 = Function("Kostomlaty Z", 1, 3.096)
filepath = "C:/KnoB/Vyroba_bastleni_realne_projekty/Jindra Akcelerometr/260629 TEST Kostomlaty/Kostomlaty-3_096Hz-0_0238HPF.txt"
fce2.load_from_file(filepath, "y", left_crop=12)
fce2.remove_mean()
fce2.fft()


p = Processor(fce, method=1)
p.run()
p2 = Processor(fce2, method=1)
p2.run()


fce3 = Function("Cosine model Kostomlaty", fce2.time, fce2.fs)
fce3.load_from_processor(p2, gon_f="cosine")
fce3.fft()

fce4 = Function("Sines 4", fce.time, fce.fs)
fce4.load_from_processor(p, gon_f="cosine")
fce4.fft()

c = Comparison()
c.add_function(fce2)
c.add_function(fce3)

# c.add_function(fce)
# c.add_function(fce4)
c.plot_all(to_one=True)


