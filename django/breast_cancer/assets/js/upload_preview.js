
(function ($) {
  $.extend({
    uploadPreview : function (options) {

      // Options + Defaults
      var settings = $.extend({
        input_field: ".image-input",
        preview_box: ".image-preview",
        label_field: ".image-label",
        label_default: "Choose File",
        label_selected: "Change File",
        no_label: false,
        success_callback : null,
        initial_image : "none",
        filesize : 1024000,
      }, options);
      
      // Check if FileReader is available
      if (window.File && window.FileList && window.FileReader) {
        if (typeof($(settings.input_field)) !== 'undefined' && $(settings.input_field) !== null) {
          $(settings.input_field).change(function() {
            var files = this.files;
            var input_image = $(settings.input_field)
            if (files.length > 0) {

              var file = files[0];
              if (file.size < settings.filesize){
                var reader = new FileReader();

              // Load file
              reader.addEventListener("load",function(event) {
                var loadedFile = event.target;

                // Check format
                if (file.type.match('image')) {
                  // Image
                  $(settings.preview_box).css("background-image", "url("+loadedFile.result+")");
                  $(settings.preview_box).css("background-size", "cover");
                  $(settings.preview_box).css("background-position", "center center");
                } else if (file.type.match('audio')) {
                  // Audio
                  $(settings.preview_box).html("<audio controls><source src='" + loadedFile.result + "' type='" + file.type + "' />Your browser does not support the audio element.</audio>");
                } else {
                  input_image.replaceWith(input_image.val('').clone(true));
                  alert("This file type is not supported yet.");
                  
                }
              });

              if (settings.no_label == false) {
                // Change label
                $(settings.label_field).html(settings.label_selected);
              }

              // Read the file
              reader.readAsDataURL(file);

              // Success callback function call
              if(settings.success_callback) {
                settings.success_callback();
              }
            }
            else {
              input_image.replaceWith(input_image.val('').clone(true));
              alert("File Size must be less than "+settings.filesize/1000+" MB");
              
            }

          } else {
            if (settings.no_label == false) {
                // Change label
                $(settings.label_field).html(settings.label_default);
              }

              // Clear background
              $(settings.preview_box).css("background-image", settings.initial_image);
              $(settings.preview_box).css("background-size", "cover");
              $(settings.preview_box).css("background-position", "center center");
              // Remove Audio
              $(settings.preview_box + " audio").remove();
            }
          });
        }
      } else {
        alert("You need a browser with file reader support, to use this form properly.");
        input_image.replaceWith(input_image.val('').clone(true));
        return false;
      }
    }
    
  });
})(jQuery);
