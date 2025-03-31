
$(document).ready(function() {
    // Handle form submission
    $('#loan-form').on('submit', function(e) {
        e.preventDefault();
        
        // Show loading state
        $('#result-container').html('<div class="alert alert-info">Processing your application...</div>');
        
        // Send form data to server
        $.ajax({
            url: '/predict',
            method: 'POST',
            data: $(this).serialize(),
            success: function(response) {
                // Update result container
                $('#result-container').html(
                    '<div class="alert ' + response.class + '">' + response.message + '</div>'
                );
            },
            error: function() {
                // Handle error
                $('#result-container').html(
                    '<div class="alert alert-danger">Error processing your request. Please try again.</div>'
                );
            }
        });
    });
    
    // Handle train button click
    $('#train-button').on('click', function() {
        // Show loading state
        $(this).html('Training...').prop('disabled', true);
        
        // Send request to train models
        $.ajax({
            url: '/train',
            method: 'GET',
            success: function(response) {
                if (response.status === 'success') {
                    // Redirect to main page
                    window.location.href = response.redirect;
                } else {
                    // Show error
                    alert('Error: ' + response.message);
                    $('#train-button').html('Train Models').prop('disabled', false);
                }
            },
            error: function() {
                // Handle error
                alert('Error training models. Please try again.');
                $('#train-button').html('Train Models').prop('disabled', false);
            }
        });
    });
});
