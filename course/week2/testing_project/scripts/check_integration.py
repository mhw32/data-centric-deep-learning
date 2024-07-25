from os.path import join
from testing.integration import MNISTIntegrationTest
from testing.paths import IMAGE_DIR


def main():
    '''Check if examples are found for the integration test.
    '''
    test_dir = join(IMAGE_DIR, "integration")
    print(test_dir)
    try:
        dataset = MNISTIntegrationTest(test_dir)
        num_examples = len(dataset.paths)
        print(f'# of examples: {num_examples}')
    except:
        print('Something went wrong? Double check your files.')


if __name__ == "__main__":
    main()
