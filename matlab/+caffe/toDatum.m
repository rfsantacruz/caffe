function datum = toDatum( varargin )
%TODATUM encode HxWxC rgb image and label to caffe protobuf WxHxC BGR.

CHECK(nargin > 0, ['usage: '...
    'datum  = toDatum( image, label )']); 
image = varargin{1};
label = varargin{2};
datum = caffe_('to_datum', image, label);

end

