import matplotlib.pyplot as plt

def savefig(file_name):
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close('all')
    print(f'Figure saved to {file_name}')
