<?php 
if(!class_exists('element_gva_gmap')):
   class element_gva_gmap{
      public function render_form(){
         $blocks = gavias_content_builder_get_blocks_options();
         $blocks[''] = '-- None --';
         $fields = array(
            'type' => 'gsc_gmap',
            'title' => t('Google Map'),
            'size' => 3,
            'fields' => array(
               array(
                  'id'     => 'title',
                  'type'      => 'text',
                  'title'  => t('Title for Admin'),
                  'admin'     => true
               ),
               array(
                  'id'           => 'map_type',
                  'type'         => 'select',
                  'title'        => t('Map Type'),
                  'options'   => array(
                     'ROADMAP'      => 'ROADMAP',
                     'HYBRID'       => 'HYBRID',
                     'SATELLITE'    => 'SATELLITE',
                     'TERRAIN'      => 'TERRAIN'
                  )
               ), 
               array(
                  'id'        => 'link',
                  'type'      => 'text',
                  'title'     => t('Latitude, Longitude for map'),
                  'desc'         => 'eg: 21.0173222,105.78405279999993',
               ),
               array(
                  'id'           => 'height',
                  'type'         => 'text',
                  'title'        => 'Map height',
                  'desc'         => 'Enter map height (in pixels or leave empty for responsive map), eg: 400px',
                  'std'          => '400px'
               ),
               array(
                 'id'        => 'info',
                 'type'      => 'info',
                 'title'      => 'Popup Block Setting'
               ),
               array(
                  'id'        => 'popup_title',
                  'type'      => 'text',
                  'title'     => t('Title Popup')
               ),
               array(
                  'id'        => 'popup_content',
                  'type'      => 'text',
                  'title'     => t('Content Popup')
               ),
               array(
                  'id'        => 'popup_block',
                  'type'      => 'select',
                  'title'     => t('Block Popup'),
                  'options'   => $blocks,
                  'class'     => 'change_value_admin'
               ),
               array(
                  'id'        => 'el_class',
                  'type'      => 'text',
                  'title'     => t('Extra class name'),
                  'desc'      => t('Style particular content element differently - add a class name and refer to it in custom CSS.'),
               ),   
               array(
                  'id'        => 'animate',
                  'type'      => 'select',
                  'title'     => t('Animation'),
                  'desc'      => t('Entrance animation for element'),
                  'options'   => gavias_content_builder_animate(),
                  'class'     => 'width-1-2'
               ), 
               array(
                  'id'        => 'animate_delay',
                  'type'      => 'select',
                  'title'     => t('Animation Delay'),
                  'options'   => gavias_content_builder_delay_aos(),
                  'desc'      => '0 = default',
                  'class'     => 'width-1-2'
               ), 
            ),                                     
         );
         return $fields;
      }

      public static function render_content( $attr = array(), $content = '' ){
         global $base_url;
         extract(gavias_merge_atts(array(
            'title'              => '',
            'map_type'           => 'ROADMAP',
            'link'               => '',
            'height'             => '',
            'animate_delay'      => '',
            'content'            => '',
            'desc'               => '',
            'popup_title'        => '',
            'popup_block'        => '',
            'el_class'           => '',
            'animate'            => '',

         ), $attr));

         $zoom = 14;
         $bubble = true;
         $_id = gavias_content_builder_makeid();
         if($animate) $el_class .= ' wow ' . $animate; 
         $style = '[{"featureType":"water","elementType":"geometry","stylers":[{"color":"#e9e9e9"},{"lightness":17}]},{"featureType":"landscape","elementType":"geometry","stylers":[{"color":"#f5f5f5"},{"lightness":20}]},{"featureType":"road.highway","elementType":"geometry.fill","stylers":[{"color":"#ffffff"},{"lightness":17}]},{"featureType":"road.highway","elementType":"geometry.stroke","stylers":[{"color":"#ffffff"},{"lightness":29},{"weight":0.2}]},{"featureType":"road.arterial","elementType":"geometry","stylers":[{"color":"#ffffff"},{"lightness":18}]},{"featureType":"road.local","elementType":"geometry","stylers":[{"color":"#ffffff"},{"lightness":16}]},{"featureType":"poi","elementType":"geometry","stylers":[{"color":"#f5f5f5"},{"lightness":21}]},{"featureType":"poi.park","elementType":"geometry","stylers":[{"color":"#dedede"},{"lightness":21}]},{"elementType":"labels.text.stroke","stylers":[{"visibility":"on"},{"color":"#ffffff"},{"lightness":16}]},{"elementType":"labels.text.fill","stylers":[{"saturation":36},{"color":"#333333"},{"lightness":40}]},{"elementType":"labels.icon","stylers":[{"visibility":"off"}]},{"featureType":"transit","elementType":"geometry","stylers":[{"color":"#f2f2f2"},{"lightness":19}]},{"featureType":"administrative","elementType":"geometry.fill","stylers":[{"color":"#fefefe"},{"lightness":20}]},{"featureType":"administrative","elementType":"geometry.stroke","stylers":[{"color":"#fefefe"},{"lightness":17},{"weight":1.2}]}]';
         $google_api = 'AIzaSyAvOhtbl4wK5vOq3YZuc-gkldJTxI13eY4';
         if(theme_get_setting('google_api_key')){
            $google_api = theme_get_setting('google_api_key');
         }

       ob_start();
         ?>
         <script src="//maps.googleapis.com/maps/api/js?key=<?php print $google_api ?>&ver=4.9.8"></script>
         <script src="<?php print (base_path() . \Drupal::service('extension.list.theme')->getPath('krowd')) ?>/vendor/gmap3.js"></script>
         <script src="<?php print (base_path() . \Drupal::service('extension.list.theme')->getPath('krowd')) ?>/vendor/jquery.ui.map.min.js"></script>

         <div class="gsc-map <?php print $el_class ?>" <?php print gavias_content_builder_print_animate_wow('', $animate_delay) ?>>
            <div id="map_canvas_<?php echo $_id; ?>" class="map_canvas" style="width:100%; height:<?php echo $height; ?>;"></div>
            <?php if($popup_block){ ?>
               <div class="content-inner clearfix">
                  <div class="link-close-content"><a href="#"><i class="fa fa-times"></i></a></div>
                  <div class="content-inner-inner">
                     <?php if($popup_title){ ?>
                        <h2 class="title"><?php echo $popup_title ?></h2>
                     <?php } ?>
                     <div class="content-block">
                        <?php echo  gavias_content_builder_render_block($popup_block); ?>
                     </div>
                  </div>
               </div>
               <div class="link-open-content"><a href="#"><i class="fa fa-map"></i></a></div>
            <?php } ?>

         </div>
         <script type="text/javascript">
            jQuery(document).ready(function($) {
               var stmapdefault = '<?php echo $link; ?>';
               var marker = {position:stmapdefault};
               var content = '<?php print $content ?>';
           
               jQuery('#map_canvas_<?php echo $_id; ?>').gmap({
                  'scrollwheel':false,
                  'zoom': <?php echo  $zoom;  ?>  ,
                  'center': stmapdefault,
                  'mapTypeId':google.maps.MapTypeId.<?php echo ( $map_type ); ?>,
                  'styles': <?php echo $style; ?>,
                  'callback': function() {
                     var self = this;
                     self.addMarker(marker).click(function(){
                        if(content){
                           self.openInfoWindow({'content': content}, self.instance.markers[0]);
                        }                     
                     });
                  },
                  panControl: true
               });
            });
         </script>
         <?php return ob_get_clean() ?>
      <?php
      }

   }
 endif;  



