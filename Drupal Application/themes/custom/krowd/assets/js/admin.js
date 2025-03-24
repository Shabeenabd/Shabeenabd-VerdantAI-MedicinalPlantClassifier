 (function ($) {
  $(document).ready(function(){
    var initCodeMirror = false;
    var myTextarea = $('#edit-customize-css');
    $('a[href="#edit-css-customize"]').click(function(){
      if(!initCodeMirror){
        var editor = CodeMirror.fromTextArea(document.getElementById("edit-customize-css"), {
          lineNumbers: true,
          mode: "text/css",
        });
      }
      initCodeMirror = true;
    })
  })
})(jQuery);

