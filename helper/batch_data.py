# def batch_parameters_for_devices(parameter_list, num_devices):
#     """
#     Organize parameters into batches for processing across devices.

#     Args:
#     parameter_list (list): The list of different parameters to batch. Each element in this list 
#                            is a set of parameters that you would use in a single function call.
#     num_devices (int): The number of available devices for parallel processing.

#     Returns:
#     list: A list of batches, each batch is a list of parameters sized for the number of devices.
#     """

#     # Calculate the number of batches needed based on the number of devices
#     num_full_batches, remainder = divmod(len(parameter_list), num_devices)
#     total_batches = num_full_batches + (1 if remainder > 0 else 0)

#     # Initialize the list of batches
#     batches = []

#     for batch_num in range(total_batches):
#         # Get the start and end index for each batch
#         start_index = batch_num * num_devices
#         end_index = start_index + num_devices

#         # If we are on the last batch and it's not full, adjust the end index
#         if batch_num == total_batches - 1 and remainder > 0:
#             end_index = start_index + remainder

#         # Extract the batch from the parameter list
#         batch = parameter_list[start_index:end_index]

#         # If the batch is not full, pad it with the last element
#         if len(batch) < num_devices:
#             batch.extend([batch[-1]] * (num_devices - len(batch)))

#         batches.append(batch)

#     return batches


def batch_parameters_for_devices(parameter_list, num_devices):
    """
    Organize parameters into batches for processing across devices.

    Args:
    parameter_list (list): List of parameter sets for function calls.
    num_devices (int): Number of available devices for parallel processing.

    Returns:
    list: A list of batches, each a list of parameter sets corresponding to the number of devices.
    """

    # Create full batches
    batches = [parameter_list[i:i + num_devices] for i in range(0, len(parameter_list), num_devices)]

    # Check if the last batch is incomplete
    last_batch = batches[-1] if batches else []
    if len(last_batch) < num_devices:
        padding = [None] * (num_devices - len(last_batch))  # Create 'empty' parameter sets
        last_batch.extend(padding)  # Add to the last batch
        batches[-1] = last_batch  # Update the last batch

    return batches
