import matplotlib.pyplot as plt


def line_plot(xlim, ylim, zlim):
    def user_postprocessor(self):
        x0 = self.coordinates[0]
        y0 = self.coordinates[1]
        z0 = self.coordinates[2]
        color_map = plt.get_cmap("tab10")
        c0 = color_map(0)
        c1 = color_map(1)
        for i in range(len(self.time)):
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_title('Time ' + str(self.time[i]))
            x = x0 + self.displacement[i][0]
            y = y0 + self.displacement[i][1]
            z = z0 + self.displacement[i][2]
            ax.plot3D(x0, y0, z0, '-', linewidth=6.0, color=c0, alpha=0.5)
            ax.plot3D(z0, y0, z0, '.-', color=c0)
            ax.plot3D(x, y, z, '-', linewidth=6.0, color=c1, alpha=0.5)
            ax.plot3D(x, y, z, '.-', color=c1)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            plt.show()
        
        plt.show()
    return user_postprocessor