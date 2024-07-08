class BaseTest:

  def get_dataloader(self):
    """To be overloaded by a child class.

    Expected to return a `torch.utils.data.DataLoader` instance.
    """
    raise NotImplementedError

  def test(self):
    """To be overloaded by a child class.

    Take actions to execute the test. Expected to return a 
    score indicating model performance on test.
    """
    raise NotImplementedError
