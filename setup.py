from setuptools import setup, find_packages

setup(
    name="iterative_learning_NMPC",
    version="0.1",
    packages=find_packages(include=["Behavior_Cloning", "Behavior_Cloning.*",
                                     "mpc_controller", "mpc_controller.*",
                                     "DAgger", "DAgger.*",
                                     "contact_tamp", "contact_tamp.*"]),
    install_requires=[
        # you can list other packages if needed
    ],
)
