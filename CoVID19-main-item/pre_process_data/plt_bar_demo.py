'''
绘制柱状图的demo
'''

np.random.seed(543)
data = np.random.randint(0,20,20)
x = np.arange(1,21,1)
print(f'data:{data}')
print(f'x:{np.arange(0,20,2)}')
plt.bar(np.arange(1,21,1), data, align='center', alpha=0.5, tick_label=x)
plt.savefig('/private/covid/demo.png')
plt.show()
