function a = AngleWrap(a)

while(a>pi)
    a=a-2*pi;
end
while(a<-pi)
    a = a+2*pi;
end
