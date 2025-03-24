(function ($, Drupal) {
  Drupal.behaviors.scrollToResult = {
    attach: function (context, settings) {
      // Use Drupal's AJAX behavior to bind to the form's AJAX callback.
      $('#image-upload-result', context).once('scrollToResult').each(function () {
        // Listen for the AJAX response complete event.
        $(document).ajaxComplete(function (event, xhr, settings) {
          // Ensure the AJAX callback is related to the form we care about.
          if (settings.extraData && settings.extraData._triggering_element_name === 'submit') {
            $('html, body').animate({
              scrollTop: $('#image-upload-result').offset().top
            }, 1000);
          }
        });
      });
    }
  };
})(jQuery, Drupal);
