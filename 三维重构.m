x=xlsread('D:\tensflow\中厚板三维重构模型\data2.csv',1,'B1:ZY1');
y=xlsread('D:\tensflow\中厚板三维重构模型\data2.csv',1,'A2:A467');
Z=xlsread('D:\tensflow\中厚板三维重构模型\data2.csv',1,'B2:zy467');
meshgrid(x',y);
mesh(X,Y,Z)